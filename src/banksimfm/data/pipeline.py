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
from banksimfm.data.schema import EVENT_TYPES, INTERVENTION_TYPES
from banksimfm.sim.engine import AccountState, apply_event, apply_intervention_policy


AMOUNT_BINS = np.array([-10000, 0, 25, 75, 150, 300, 600, 1200, 5000, 20000], dtype=float)
BALANCE_BINS = np.array([-10000, 0, 250, 750, 1500, 3000, 5000, 10000, 30000], dtype=float)
UTIL_BINS = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.1, 2.0], dtype=float)
GAP_BINS = np.array([0, 1, 2, 4, 7, 14, 30, 60], dtype=float)
DELTA_BINS = np.array([-10000, -2000, -1000, -500, -200, -50, 0, 50, 200, 500, 1000, 2000, 10000], dtype=float)

DISTRESS_EVENT_TYPES = {"loan_emi_missed", "failed_debit", "overdraft_event"}


@dataclass
class Encoders:
    event_type_to_id: Dict[str, int]
    event_id_to_type: Dict[int, str]
    intervention_type_to_id: Dict[str, int]
    intervention_id_to_type: Dict[int, str]


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
    intervention_type_to_id = {name: idx for idx, name in enumerate(INTERVENTION_TYPES)}
    intervention_id_to_type = {idx: name for name, idx in intervention_type_to_id.items()}
    return Encoders(
        event_type_to_id=event_type_to_id,
        event_id_to_type=event_id_to_type,
        intervention_type_to_id=intervention_type_to_id,
        intervention_id_to_type=intervention_id_to_type,
    )


def _bucketize(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    return np.digitize(values.astype(float).to_numpy(), bins=bins[1:], right=False).astype(np.int64)


def _bucketize_scalar(value: float, bins: np.ndarray) -> int:
    return int(np.digitize([float(value)], bins[1:], right=False)[0])


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
    frame["balance_delta"] = frame.groupby("customer_id")["balance_after"].diff().fillna(frame["balance_after"] - frame["balance_before"])
    frame["balance_delta_bucket"] = _bucketize(frame["balance_delta"], DELTA_BINS)
    frame["direction_flag"] = frame["amount_direction"].map({"credit": 1.0, "debit": -1.0}).fillna(0.0)
    frame["miss_flag"] = frame["event_type"].isin(["loan_emi_missed", "failed_debit"]).astype(int)
    frame["overdraft_flag"] = frame["event_type"].eq("overdraft_event").astype(int)
    frame["intervention_id"] = 0
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


def _history_to_state_from_frame(history: pd.DataFrame) -> AccountState:
    last_row = history.iloc[-1]
    missed = int(history["event_type"].isin(["loan_emi_missed"]).sum())
    failed_debits = int(history["event_type"].eq("failed_debit").sum())
    overdrafts = int(history["event_type"].eq("overdraft_event").sum())
    low_balance_streak = int((history["balance_after"] < 200).tail(5).sum())
    return AccountState(
        balance=float(last_row["balance_after"]),
        credit_limit=float(last_row["credit_limit"]),
        utilized_credit=float(last_row["credit_utilization"]) * float(last_row["credit_limit"]),
        missed_payments=missed,
        overdraft_events=overdrafts,
        failed_debits=failed_debits,
        low_balance_streak=low_balance_streak,
        due_amount=float(max(0.0, history["amount"].tail(3).mean())),
        due_in_days=int(last_row.get("days_to_next_due", 7) or 7),
        last_income_amount=float(history.loc[history["event_type"] == "salary_credit", "amount"].tail(1).iloc[0] if not history.loc[history["event_type"] == "salary_credit"].empty else 0.0),
        intervention_flag=str(last_row.get("intervention_flag", "none")),
    )


def _counterfactual_next_event(
    state: AccountState,
    intervention_type: str,
    rng: np.random.Generator,
) -> tuple[str, float, str]:
    if state.due_amount > 0 and state.due_in_days <= 2:
        coverage_factor = 0.72 if intervention_type != "none" else 0.88
        if state.balance > state.due_amount * coverage_factor:
            return "loan_emi_paid", min(max(40.0, state.due_amount * 0.8), max(40.0, state.balance * 0.18)), "loan"
        return "loan_emi_missed", max(40.0, state.due_amount), "loan"
    if state.balance < 150:
        return "failed_debit", max(35.0, state.due_amount or 70.0), "credit"
    if state.credit_utilization > 0.9:
        return "credit_card_payment_made", min(max(50.0, state.utilized_credit * 0.18), max(50.0, state.balance * 0.15)), "credit"

    choices = [
        ("salary_credit", max(1800.0, state.last_income_amount or 2500.0), "income"),
        ("grocery_spend", float(rng.uniform(25, 120)), "daily_needs"),
        ("card_spend", float(rng.uniform(40, 180)), "credit"),
        ("utility_payment", float(rng.uniform(45, 110)), "utilities"),
        ("transportation_spend", float(rng.uniform(10, 40)), "transport"),
        ("transfer_in", float(rng.uniform(60, 220)), "support_network"),
    ]
    if intervention_type == "temporary_overdraft_buffer":
        weights = [0.14, 0.24, 0.18, 0.1, 0.14, 0.2]
    elif intervention_type == "installment_restructure":
        weights = [0.12, 0.28, 0.18, 0.08, 0.16, 0.18]
    elif intervention_type == "reminder":
        weights = [0.12, 0.24, 0.18, 0.12, 0.16, 0.18]
    else:
        weights = [0.12, 0.28, 0.2, 0.1, 0.18, 0.12]
    idx = int(rng.choice(np.arange(len(choices)), p=np.asarray(weights) / np.sum(weights)))
    return choices[idx]


def _days_to_first_distress(customer_frame: pd.DataFrame, history_end_idx: int) -> float:
    history_end_ts = pd.to_datetime(customer_frame.iloc[history_end_idx]["event_timestamp"])
    future = customer_frame.iloc[history_end_idx + 1:]
    for _, row in future.iterrows():
        row_ts = pd.to_datetime(row["event_timestamp"])
        if (row_ts - history_end_ts).days > 30:
            break
        if row["event_type"] in DISTRESS_EVENT_TYPES:
            return float((row_ts - history_end_ts).days)
    return -1.0


class CustomerWindowDataset(Dataset):
    """Shared customer-history windows for both sequence models."""

    def __init__(self, frame: pd.DataFrame, context_length: int, encoders: Encoders, intervention_aug_rate: float = 0.15, seed: int = 7):
        self.context_length = context_length
        self.encoders = encoders
        self.intervention_aug_rate = intervention_aug_rate
        self.rng = np.random.default_rng(seed)
        self.samples = self._build_samples(frame)

    def _history_tensors(self, history: pd.DataFrame, intervention_type: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_cols = [
            "event_type_id",
            "amount_bucket",
            "balance_bucket",
            "util_bucket",
            "time_gap_bucket",
            "intervention_id",
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

        history = history.copy()
        history["intervention_id"] = self.encoders.intervention_type_to_id[intervention_type]
        seq_features = torch.zeros((self.context_length, len(feature_cols)), dtype=torch.long)
        dense_features = torch.zeros((self.context_length, len(numeric_cols)), dtype=torch.float32)
        mask = torch.zeros(self.context_length, dtype=torch.bool)
        history_offset = self.context_length - len(history)
        seq_features[history_offset:, :] = torch.tensor(history[feature_cols].to_numpy(), dtype=torch.long)
        dense_values = history[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
        dense_features[history_offset:, :] = torch.tensor(dense_values, dtype=torch.float32)
        mask[history_offset:] = True
        return seq_features, dense_features, mask

    def _base_sample(
        self,
        customer_frame: pd.DataFrame,
        history: pd.DataFrame,
        target: pd.Series,
        target_idx: int,
        intervention_type: str = "none",
        lstm_weight: float = 1.0,
    ) -> Dict[str, object]:
        seq_features, dense_features, mask = self._history_tensors(history, intervention_type)
        balance_delta = float(target["balance_after"] - history.iloc[-1]["balance_after"])
        days_to_first_distress = _days_to_first_distress(customer_frame, target_idx - 1)
        return {
            "seq_tokens": seq_features,
            "dense_features": dense_features,
            "mask": mask,
            "target_event": torch.tensor(int(target["event_type_id"]), dtype=torch.long),
            "target_amount_bucket": torch.tensor(int(target["amount_bucket"]), dtype=torch.long),
            "target_balance_delta_bucket": torch.tensor(_bucketize_scalar(balance_delta, DELTA_BINS), dtype=torch.long),
            "target_distress": torch.tensor(float(target["distress_label_30d"]), dtype=torch.float32),
            "transformer_sample_weight": torch.tensor(1.0, dtype=torch.float32),
            "lstm_sample_weight": torch.tensor(lstm_weight, dtype=torch.float32),
            "customer_id": str(target["customer_id"]),
            "history_end_timestamp": history.iloc[-1]["event_timestamp"].isoformat(),
            "days_to_first_distress": torch.tensor(days_to_first_distress, dtype=torch.float32),
        }

    def _counterfactual_sample(self, history: pd.DataFrame, customer_frame: pd.DataFrame, target_idx: int) -> Dict[str, object]:
        intervention_type = str(self.rng.choice(INTERVENTION_TYPES[1:]))
        current_state = _history_to_state_from_frame(history)
        adjusted_state = apply_intervention_policy(current_state, intervention_type)

        history_cf = history.copy()
        last_idx = history_cf.index[-1]
        history_cf.loc[last_idx, "balance_after"] = adjusted_state.balance
        history_cf.loc[last_idx, "credit_utilization"] = adjusted_state.credit_utilization
        history_cf.loc[last_idx, "days_to_next_due"] = adjusted_state.due_in_days
        history_cf.loc[last_idx, "intervention_flag"] = intervention_type
        history_cf.loc[last_idx, "balance_bucket"] = _bucketize_scalar(adjusted_state.balance, BALANCE_BINS)
        history_cf.loc[last_idx, "util_bucket"] = _bucketize_scalar(adjusted_state.credit_utilization, UTIL_BINS)

        event_type, amount, category = _counterfactual_next_event(adjusted_state, intervention_type, self.rng)
        next_state = apply_event(adjusted_state, event_type, amount)
        target = pd.Series(
            {
                "customer_id": history.iloc[-1]["customer_id"],
                "event_type_id": self.encoders.event_type_to_id[event_type],
                "amount_bucket": _bucketize_scalar(amount, AMOUNT_BINS),
                "balance_after": next_state.balance,
                "distress_label_30d": int(
                    event_type in DISTRESS_EVENT_TYPES
                    or next_state.balance < 0
                    or next_state.credit_utilization > 0.95
                    or next_state.failed_debits > adjusted_state.failed_debits
                    or next_state.missed_payments > adjusted_state.missed_payments
                ),
            }
        )
        return self._base_sample(customer_frame, history_cf, target, target_idx, intervention_type=intervention_type, lstm_weight=0.0)

    def _build_samples(self, frame: pd.DataFrame) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        for _, customer_frame in frame.groupby("customer_id"):
            customer_frame = customer_frame.sort_values("event_timestamp").reset_index(drop=True)
            for idx in range(1, len(customer_frame)):
                history = customer_frame.iloc[max(0, idx - self.context_length):idx].copy()
                target = customer_frame.iloc[idx]
                samples.append(self._base_sample(customer_frame, history, target, idx))
                if len(history) >= 5 and self.rng.random() < self.intervention_aug_rate:
                    samples.append(self._counterfactual_sample(history, customer_frame, idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
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
    intervention_aug_rate = getattr(config.model, "intervention_augmentation_rate", 0.15)
    datasets = {
        name: CustomerWindowDataset(
            split_frame,
            context_length=config.model.context_length,
            encoders=encoders,
            intervention_aug_rate=intervention_aug_rate if name == "train" else 0.0,
            seed=config.data.seed,
        )
        for name, split_frame in split_frames.items()
    }
    bundle = DemoBundle(events=prepared, customers=customers, splits=split_frames, encoders=encoders)
    return bundle, datasets


def load_or_create_demo_bundle(config: ProjectConfig | None = None, force_regenerate: bool = False) -> DemoBundle:
    config = config or default_config()
    artifacts_dir = config.artifacts_dir
    events_path = artifacts_dir / "demo_events.csv"
    customers_path = artifacts_dir / "demo_customers.csv"
    metadata_path = artifacts_dir / "demo_metadata.json"

    if not force_regenerate and events_path.exists() and customers_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        required_customer_cols = {"income_band", "employment_type", "region", "risk_segment"}
        if "intervention_type_to_id" in metadata:
            customers = pd.read_csv(customers_path)
            if required_customer_cols.issubset(set(customers.columns)):
                events = pd.read_csv(events_path, parse_dates=["event_timestamp"])
                encoders = Encoders(
                    event_type_to_id=metadata["event_type_to_id"],
                    event_id_to_type={int(key): value for key, value in metadata["event_id_to_type"].items()},
                    intervention_type_to_id=metadata["intervention_type_to_id"],
                    intervention_id_to_type={int(key): value for key, value in metadata["intervention_id_to_type"].items()},
                )
                prepared = _prepare_features(events, encoders)
                split_frames = {
                    name: prepared[prepared["customer_id"].isin(ids)].reset_index(drop=True)
                    for name, ids in metadata["splits"].items()
                }
                return DemoBundle(events=prepared, customers=customers, splits=split_frames, encoders=encoders)

    bundle, _ = build_datasets(config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle.events.to_csv(events_path, index=False)
    bundle.customers.to_csv(customers_path, index=False)
    metadata = {
        "event_type_to_id": bundle.encoders.event_type_to_id,
        "event_id_to_type": bundle.encoders.event_id_to_type,
        "intervention_type_to_id": bundle.encoders.intervention_type_to_id,
        "intervention_id_to_type": bundle.encoders.intervention_id_to_type,
        "splits": {
            name: sorted(frame["customer_id"].unique().tolist())
            for name, frame in bundle.splits.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return bundle
