"""Training utilities for BankSimFM models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from banksimfm.config import ProjectConfig, default_config
from banksimfm.data.pipeline import AMOUNT_BINS, BALANCE_BINS, GAP_BINS, UTIL_BINS, build_datasets
from banksimfm.models.baseline import LSTMDistressModel
from banksimfm.models.transformer import CausalEventTransformer
from banksimfm.runtime import resolve_device


@dataclass
class TrainingArtifacts:
    metrics: Dict[str, Dict[str, float]]
    model_paths: Dict[str, Path]


def _build_dataloaders(datasets: Dict[str, torch.utils.data.Dataset], batch_size: int) -> Dict[str, DataLoader]:
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }


def _compute_binary_metrics(targets: Iterable[float], probs: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(list(targets)).astype(int)
    y_prob = np.asarray(list(probs))
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _evaluate_transformer(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            seq_tokens = batch["seq_tokens"].to(device)
            mask = batch["mask"].to(device)
            _, distress_logits = model(seq_tokens, mask)
            probs.extend(torch.sigmoid(distress_logits).cpu().tolist())
            targets.extend(batch["target_distress"].tolist())
    return _compute_binary_metrics(targets, probs)


def _evaluate_lstm(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            dense_features = batch["dense_features"].to(device)
            mask = batch["mask"].to(device)
            logits = model(dense_features, mask)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            targets.extend(batch["target_distress"].tolist())
    return _compute_binary_metrics(targets, probs)


def train_models(config: ProjectConfig | None = None) -> TrainingArtifacts:
    config = config or default_config()
    bundle, datasets = build_datasets(config)
    dataloaders = _build_dataloaders(datasets, config.model.batch_size)
    device = resolve_device()

    transformer = CausalEventTransformer(
        vocab_size=len(bundle.encoders.event_type_to_id) + 1,
        bucket_sizes={
            "amount": len(AMOUNT_BINS),
            "balance": len(BALANCE_BINS),
            "util": len(UTIL_BINS),
            "gap": len(GAP_BINS),
        },
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    ).to(device)

    lstm = LSTMDistressModel(
        input_size=dataloaders["train"].dataset[0]["dense_features"].shape[-1],
        hidden_size=max(64, config.model.hidden_size // 2),
        dropout=config.model.dropout,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    transformer_optimizer = torch.optim.Adam(transformer.parameters(), lr=config.model.learning_rate)
    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=config.model.learning_rate)

    best_transformer_auc = -1.0
    best_lstm_auc = -1.0
    patience_transformer = config.model.patience
    patience_lstm = config.model.patience

    artifacts_dir = config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transformer_path = artifacts_dir / "transformer.pt"
    lstm_path = artifacts_dir / "lstm.pt"

    for _ in range(config.model.max_epochs):
        transformer.train()
        lstm.train()
        for batch in dataloaders["train"]:
            seq_tokens = batch["seq_tokens"].to(device)
            dense_features = batch["dense_features"].to(device)
            mask = batch["mask"].to(device)
            target_event = batch["target_event"].to(device)
            target_distress = batch["target_distress"].to(device)

            transformer_optimizer.zero_grad()
            next_event_logits, distress_logits = transformer(seq_tokens, mask)
            transformer_loss = ce_loss(next_event_logits, target_event) + bce_loss(distress_logits, target_distress)
            transformer_loss.backward()
            transformer_optimizer.step()

            lstm_optimizer.zero_grad()
            lstm_logits = lstm(dense_features, mask)
            lstm_loss = bce_loss(lstm_logits, target_distress)
            lstm_loss.backward()
            lstm_optimizer.step()

        transformer_val = _evaluate_transformer(transformer, dataloaders["val"], device)
        lstm_val = _evaluate_lstm(lstm, dataloaders["val"], device)

        if transformer_val["auc"] > best_transformer_auc:
            best_transformer_auc = transformer_val["auc"]
            torch.save(transformer.state_dict(), transformer_path)
            patience_transformer = config.model.patience
        else:
            patience_transformer -= 1

        if lstm_val["auc"] > best_lstm_auc:
            best_lstm_auc = lstm_val["auc"]
            torch.save(lstm.state_dict(), lstm_path)
            patience_lstm = config.model.patience
        else:
            patience_lstm -= 1

        if patience_transformer <= 0 and patience_lstm <= 0:
            break

    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))

    metrics = {
        "transformer": {
            "validation": _evaluate_transformer(transformer, dataloaders["val"], device),
            "test": _evaluate_transformer(transformer, dataloaders["test"], device),
        },
        "lstm": {
            "validation": _evaluate_lstm(lstm, dataloaders["val"], device),
            "test": _evaluate_lstm(lstm, dataloaders["test"], device),
        },
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return TrainingArtifacts(metrics=metrics, model_paths={"transformer": transformer_path, "lstm": lstm_path})
