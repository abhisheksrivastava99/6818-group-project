"""Training utilities for BankSimFM models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        "threshold": float(threshold),
    }


def _collect_transformer_outputs(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
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
    return probs, targets


def _collect_lstm_outputs(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
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
    return probs, targets


def _positive_class_weight(dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for batch in dataloader:
        target_distress = batch["target_distress"]
        positives += float(target_distress.sum().item())
        total += float(target_distress.numel())
    negatives = max(1.0, total - positives)
    positives = max(1.0, positives)
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def _select_best_threshold(targets: Iterable[float], probs: Iterable[float]) -> float:
    y_true = np.asarray(list(targets)).astype(int)
    y_prob = np.asarray(list(probs))
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.2, 0.8, 25):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold


def _format_metric_block(metrics: Dict[str, float]) -> str:
    return (
        f"auc={metrics['auc']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"threshold={metrics['threshold']:.3f}"
    )


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
    pos_weight = _positive_class_weight(dataloaders["train"], device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

    print(
        f"Training on device={device} "
        f"train_samples={len(dataloaders['train'].dataset)} "
        f"val_samples={len(dataloaders['val'].dataset)} "
        f"test_samples={len(dataloaders['test'].dataset)} "
        f"pos_weight={pos_weight.item():.4f}"
    )
    print(
        f"Config | epochs={config.model.max_epochs} "
        f"patience={config.model.patience} "
        f"batch_size={config.model.batch_size} "
        f"lr={config.model.learning_rate} "
        f"context_length={config.model.context_length}"
    )

    for epoch_idx in range(config.model.max_epochs):
        transformer.train()
        lstm.train()
        transformer_loss_total = 0.0
        lstm_loss_total = 0.0
        batch_count = 0
        first_batch_logged = False
        for batch in dataloaders["train"]:
            seq_tokens = batch["seq_tokens"].to(device)
            dense_features = batch["dense_features"].to(device)
            mask = batch["mask"].to(device)
            target_event = batch["target_event"].to(device)
            target_distress = batch["target_distress"].to(device)

            if not first_batch_logged:
                positive_rate = float(target_distress.mean().item())
                active_tokens = float(mask.sum(dim=1).float().mean().item())
                print(
                    f"Epoch {epoch_idx + 1}/{config.model.max_epochs} | "
                    f"first_batch_size={target_distress.shape[0]} "
                    f"first_batch_positive_rate={positive_rate:.4f} "
                    f"avg_active_tokens={active_tokens:.1f}"
                )
                first_batch_logged = True

            transformer_optimizer.zero_grad()
            next_event_logits, distress_logits = transformer(seq_tokens, mask)
            transformer_loss = ce_loss(next_event_logits, target_event) + bce_loss(distress_logits, target_distress)
            transformer_loss.backward()
            transformer_optimizer.step()
            transformer_loss_total += float(transformer_loss.item())

            lstm_optimizer.zero_grad()
            lstm_logits = lstm(dense_features, mask)
            lstm_loss = bce_loss(lstm_logits, target_distress)
            lstm_loss.backward()
            lstm_optimizer.step()
            lstm_loss_total += float(lstm_loss.item())
            batch_count += 1

        transformer_val_probs, transformer_val_targets = _collect_transformer_outputs(transformer, dataloaders["val"], device)
        lstm_val_probs, lstm_val_targets = _collect_lstm_outputs(lstm, dataloaders["val"], device)
        transformer_val = _compute_binary_metrics(transformer_val_targets, transformer_val_probs, threshold=0.5)
        lstm_val = _compute_binary_metrics(lstm_val_targets, lstm_val_probs, threshold=0.5)

        transformer_improved = transformer_val["auc"] > best_transformer_auc
        if transformer_val["auc"] > best_transformer_auc:
            best_transformer_auc = transformer_val["auc"]
            torch.save(transformer.state_dict(), transformer_path)
            patience_transformer = config.model.patience
        else:
            patience_transformer -= 1

        lstm_improved = lstm_val["auc"] > best_lstm_auc
        if lstm_val["auc"] > best_lstm_auc:
            best_lstm_auc = lstm_val["auc"]
            torch.save(lstm.state_dict(), lstm_path)
            patience_lstm = config.model.patience
        else:
            patience_lstm -= 1

        print(
            f"Epoch {epoch_idx + 1}/{config.model.max_epochs} summary | "
            f"transformer_loss={transformer_loss_total / max(1, batch_count):.4f} "
            f"lstm_loss={lstm_loss_total / max(1, batch_count):.4f} | "
            f"transformer_val_auc={transformer_val['auc']:.4f} "
            f"transformer_val_f1@0.5={transformer_val['f1']:.4f} "
            f"{'saved' if transformer_improved else f'patience={patience_transformer}'} | "
            f"lstm_val_auc={lstm_val['auc']:.4f} "
            f"lstm_val_f1@0.5={lstm_val['f1']:.4f} "
            f"{'saved' if lstm_improved else f'patience={patience_lstm}'}"
        )
        print(f"  Transformer val @0.5 | {_format_metric_block(transformer_val)}")
        print(f"  LSTM val @0.5        | {_format_metric_block(lstm_val)}")
        print(
            f"  Best so far | transformer_auc={best_transformer_auc:.4f} "
            f"lstm_auc={best_lstm_auc:.4f}"
        )

        if patience_transformer <= 0 and patience_lstm <= 0:
            print("Early stopping triggered for both models.")
            break

    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))

    transformer_val_probs, transformer_val_targets = _collect_transformer_outputs(transformer, dataloaders["val"], device)
    lstm_val_probs, lstm_val_targets = _collect_lstm_outputs(lstm, dataloaders["val"], device)
    transformer_threshold = _select_best_threshold(transformer_val_targets, transformer_val_probs)
    lstm_threshold = _select_best_threshold(lstm_val_targets, lstm_val_probs)
    transformer_test_probs, transformer_test_targets = _collect_transformer_outputs(transformer, dataloaders["test"], device)
    lstm_test_probs, lstm_test_targets = _collect_lstm_outputs(lstm, dataloaders["test"], device)

    metrics = {
        "transformer": {
            "validation": _compute_binary_metrics(transformer_val_targets, transformer_val_probs, threshold=transformer_threshold),
            "test": _compute_binary_metrics(transformer_test_targets, transformer_test_probs, threshold=transformer_threshold),
        },
        "lstm": {
            "validation": _compute_binary_metrics(lstm_val_targets, lstm_val_probs, threshold=lstm_threshold),
            "test": _compute_binary_metrics(lstm_test_targets, lstm_test_probs, threshold=lstm_threshold),
        },
    }
    print(
        f"Selected thresholds | transformer={transformer_threshold:.3f} "
        f"lstm={lstm_threshold:.3f}"
    )
    print(f"Final transformer validation | {_format_metric_block(metrics['transformer']['validation'])}")
    print(f"Final transformer test       | {_format_metric_block(metrics['transformer']['test'])}")
    print(f"Final LSTM validation        | {_format_metric_block(metrics['lstm']['validation'])}")
    print(f"Final LSTM test              | {_format_metric_block(metrics['lstm']['test'])}")
    print(json.dumps(metrics, indent=2))
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return TrainingArtifacts(metrics=metrics, model_paths={"transformer": transformer_path, "lstm": lstm_path})
