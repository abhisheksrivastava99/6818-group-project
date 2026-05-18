"""Preprocessing and dataset construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from banksimfm.config import ProjectConfig, default_config
from banksimfm.data.generator import generate_synthetic_dataset
from banksimfm.data.schema import EVENT_TYPES


AMOUNT_BINS = np.array([-10000, 0, 25, 75, 150, 300, 600, 1200, 5000, 20000], dtype=float)
BALANCE_BINS = np.array([-10000, 0, 250, 750, 1500, 3000, 5000, 10000, 30000], dtype=float)
UTIL_BINS = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.1, 2.0], dtype=float)
GAP_BINS = np.array([0, 1, 2, 4, 7, 14, 30, 60], dtype=float)


@dataclass
class Encoders:
    event_type_to_id: Dict[str, int]
    event_id_to_type: Dict[int, str]


@dataclass
class DemoBundle:
    events: pd.DataFrame
    customers: pd.DataFrame
    splits: Dict[str, pd.DataFrame]
    encoders: Encoders


def build_encoders() -> Encoders:
    event_type_to_id = {event_type: idx + 1 for idx, event_type in enumerate(EVENT_TYPES)}
    event_id_to_type = {idx: event_type for event_type, idx in event_type_to_id.items()}
    event_id_to_type[0] = "PAD"
    return Encoders(event_type_to_id=event_type_to_id, event_id_to_type=event_id_to_type)


def _bucketize(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    return np.digitize(values.astype(float).to_numpy(), bins=bins[1:], right=False).astype(np.int64)


def _prepare_features(events: pd.DataFrame, encoders: Encoders) -> pd.DataFrame:
    frame = events.copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    frame = frame.sort_values(["customer_id", "event_timestamp"]).reset_index(drop=True)
    frame["event_type_id"] = frame["event_type"].map(encoders.event_type_to_id).fillna(0).astype(int)
    frame["amount_bucket"] = _bucketize(frame["amount"], AMOUNT_BINS)
    frame["balance_bucket"] = _bucketize(frame["balance_after"], BALANCE_BINS)
    frame["util_bucket"] = _bucketize(frame["credit_utilization"], UTIL_BINS)
    frame["time_gap_days"] = frame.groupby("customer_id")["event_timestamp"].diff().dt.total_seconds().fillna(0) / 86400
    frame["time_gap_bucket"] = _bucketize(frame["time_gap_days"], GAP_BINS)
    frame["direction_flag"] = frame["amount_direction"].map({"credit": 1.0, "debit": -1.0}).fillna(0.0)
    frame["miss_flag"] = frame["event_type"].isin(["loan_emi_missed", "failed_debit"]).astype(int)
    frame["overdraft_flag"] = frame["event_type"].eq("overdraft_event").astype(int)
    return frame


def split_customers(customers: pd.DataFrame, seed: int) -> Dict[str, List[str]]:
    def maybe_stratify(labels: pd.Series) -> pd.Series | None:
        counts = labels.value_counts()
        if counts.empty or counts.min() < 2:
            return None
        return labels

    first_labels = maybe_stratify(customers["archetype"])
    train_ids, temp_ids = train_test_split(
        customers["customer_id"],
        test_size=0.3,
        random_state=seed,
        stratify=first_labels,
    )
    second_labels = maybe_stratify(customers.set_index("customer_id").loc[temp_ids, "archetype"])
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=seed,
        stratify=second_labels,
    )
    return {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}


class CustomerWindowDataset(Dataset):
    """Shared customer-history windows for both sequence models."""

    def __init__(self, frame: pd.DataFrame, context_length: int):
        self.context_length = context_length
        self.samples = self._build_samples(frame)

    def _build_samples(self, frame: pd.DataFrame) -> List[Dict[str, torch.Tensor]]:
        samples: List[Dict[str, torch.Tensor]] = []
        feature_cols = [
            "event_type_id",
            "amount_bucket",
            "balance_bucket",
            "util_bucket",
            "time_gap_bucket",
        ]
        numeric_cols = [
            "amount",
            "balance_after",
            "credit_utilization",
            "days_to_next_due",
            "direction_flag",
            "miss_flag",
            "overdraft_flag",
        ]

        for _, customer_frame in frame.groupby("customer_id"):
            customer_frame = customer_frame.sort_values("event_timestamp").reset_index(drop=True)
            for idx in range(1, len(customer_frame)):
                history = customer_frame.iloc[max(0, idx - self.context_length):idx]
                target = customer_frame.iloc[idx]

                seq_features = torch.zeros((self.context_length, len(feature_cols)), dtype=torch.long)
                dense_features = torch.zeros((self.context_length, len(numeric_cols)), dtype=torch.float32)
                mask = torch.zeros(self.context_length, dtype=torch.bool)

                history_offset = self.context_length - len(history)
                seq_features[history_offset:, :] = torch.tensor(history[feature_cols].to_numpy(), dtype=torch.long)
                dense_values = history[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
                dense_features[history_offset:, :] = torch.tensor(dense_values, dtype=torch.float32)
                mask[history_offset:] = True

                samples.append(
                    {
                        "seq_tokens": seq_features,
                        "dense_features": dense_features,
                        "mask": mask,
                        "target_event": torch.tensor(int(target["event_type_id"]), dtype=torch.long),
                        "target_distress": torch.tensor(float(target["distress_label_30d"]), dtype=torch.float32),
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def build_datasets(config: ProjectConfig | None = None) -> Tuple[DemoBundle, Dict[str, CustomerWindowDataset]]:
    config = config or default_config()
    events, customers = generate_synthetic_dataset(config.data)
    encoders = build_encoders()
    prepared = _prepare_features(events, encoders)
    splits = split_customers(customers, config.data.seed)

    split_frames = {
        name: prepared[prepared["customer_id"].isin(customer_ids)].reset_index(drop=True)
        for name, customer_ids in splits.items()
    }
    datasets = {
        name: CustomerWindowDataset(split_frame, context_length=config.model.context_length)
        for name, split_frame in split_frames.items()
    }
    bundle = DemoBundle(events=prepared, customers=customers, splits=split_frames, encoders=encoders)
    return bundle, datasets


def load_or_create_demo_bundle(config: ProjectConfig | None = None) -> DemoBundle:
    config = config or default_config()
    artifacts_dir = config.artifacts_dir
    events_path = artifacts_dir / "demo_events.csv"
    customers_path = artifacts_dir / "demo_customers.csv"
    metadata_path = artifacts_dir / "demo_metadata.json"

    if events_path.exists() and customers_path.exists() and metadata_path.exists():
        events = pd.read_csv(events_path, parse_dates=["event_timestamp"])
        customers = pd.read_csv(customers_path)
        metadata = json.loads(metadata_path.read_text())
        encoders = Encoders(
            event_type_to_id=metadata["event_type_to_id"],
            event_id_to_type={int(key): value for key, value in metadata["event_id_to_type"].items()},
        )
        prepared = _prepare_features(events, encoders)
        split_frames = {name: prepared[prepared["customer_id"].isin(ids)].reset_index(drop=True) for name, ids in metadata["splits"].items()}
        return DemoBundle(events=prepared, customers=customers, splits=split_frames, encoders=encoders)

    bundle, _ = build_datasets(config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle.events.to_csv(events_path, index=False)
    bundle.customers.to_csv(customers_path, index=False)
    metadata = {
        "event_type_to_id": bundle.encoders.event_type_to_id,
        "event_id_to_type": bundle.encoders.event_id_to_type,
        "splits": {
            name: sorted(frame["customer_id"].unique().tolist())
            for name, frame in bundle.splits.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return bundle
