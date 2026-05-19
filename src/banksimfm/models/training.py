"""Training utilities for BankSimFM models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from banksimfm.config import ProjectConfig, default_config
from banksimfm.data.pipeline import (
    AMOUNT_BINS,
    BALANCE_BINS,
    DELTA_BINS,
    DUE_BINS,
    DISTRESS_EVENT_TYPES,
    GAP_BINS,
    UTIL_BINS,
    build_datasets,
    compute_due_amount_feature,
)
from banksimfm.data.schema import INTERVENTION_TYPES
from banksimfm.models.baseline import LSTMDistressModel
from banksimfm.models.transformer import CausalEventTransformer
from banksimfm.runtime import resolve_device
from banksimfm.sim.engine import apply_intervention_policy
from banksimfm.sim.scenario import apply_state_to_history, decode_forecast, history_to_state


SEGMENT_FIELDS = ["archetype", "income_band", "employment_type", "region", "risk_segment"]


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
            outputs = model(seq_tokens, mask)
            probs.extend(torch.sigmoid(outputs["distress_logits"]).cpu().tolist())
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


def _direction_flag(event_type: str, amount_direction: str | None = None) -> float:
    if amount_direction is not None:
        return {"credit": 1.0, "debit": -1.0}.get(amount_direction, 0.0)
    return 1.0 if event_type in {"salary_credit", "transfer_in"} else -1.0


def _prepare_lstm_tensors(history: pd.DataFrame, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    frame = history.copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    frame = frame.sort_values("event_timestamp").tail(context_length).reset_index(drop=True)
    if "direction_flag" not in frame.columns:
        frame["direction_flag"] = [
            _direction_flag(event_type, row.get("amount_direction"))
            for event_type, (_, row) in zip(frame["event_type"], frame.iterrows())
        ]
    if "miss_flag" not in frame.columns:
        frame["miss_flag"] = frame["event_type"].isin(["loan_emi_missed", "failed_debit"]).astype(float)
    if "overdraft_flag" not in frame.columns:
        frame["overdraft_flag"] = frame["event_type"].eq("overdraft_event").astype(float)
    if "days_to_next_due" not in frame.columns:
        frame["days_to_next_due"] = 7.0
    if "due_amount_feature" not in frame.columns:
        frame["due_amount_feature"] = compute_due_amount_feature(frame)
    elif frame["due_amount_feature"].isna().any():
        frame["due_amount_feature"] = frame["due_amount_feature"].fillna(compute_due_amount_feature(frame))
    dense_features = torch.zeros((1, context_length, 8), dtype=torch.float32)
    mask = torch.zeros((1, context_length), dtype=torch.bool)
    offset = context_length - len(frame)
    dense_features[0, offset:, :] = torch.tensor(
        frame[
            [
                "amount",
                "balance_after",
                "credit_utilization",
                "due_amount_feature",
                "days_to_next_due",
                "direction_flag",
                "miss_flag",
                "overdraft_flag",
            ]
        ].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    mask[0, offset:] = True
    return dense_features, mask


def _lstm_probability_from_history(model: nn.Module, history: pd.DataFrame, device: torch.device, context_length: int) -> float:
    dense_features, mask = _prepare_lstm_tensors(history, context_length)
    with torch.no_grad():
        logits = model(dense_features.to(device), mask.to(device))
    return float(torch.sigmoid(logits).item())


def _collect_sample_records(transformer: nn.Module, lstm: nn.Module, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    transformer.eval()
    lstm.eval()
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in dataloader:
            seq_tokens = batch["seq_tokens"].to(device)
            dense_features = batch["dense_features"].to(device)
            mask = batch["mask"].to(device)
            transformer_outputs = transformer(seq_tokens, mask)
            lstm_logits = lstm(dense_features, mask)
            transformer_probs = torch.sigmoid(transformer_outputs["distress_logits"]).cpu().tolist()
            lstm_probs = torch.sigmoid(lstm_logits).cpu().tolist()
            for idx, customer_id in enumerate(batch["customer_id"]):
                records.append(
                    {
                        "customer_id": customer_id,
                        "target_distress": float(batch["target_distress"][idx].item()),
                        "transformer_prob": float(transformer_probs[idx]),
                        "lstm_prob": float(lstm_probs[idx]),
                        "days_to_first_distress": float(batch["days_to_first_distress"][idx].item()),
                    }
                )
    return pd.DataFrame(records)


def _metric_dict_for_group(df: pd.DataFrame, prob_col: str, threshold: float) -> Dict[str, float]:
    metrics = _compute_binary_metrics(df["target_distress"], df[prob_col], threshold=threshold)
    negatives = df[df["target_distress"] == 0]
    if len(negatives) == 0:
        fpr = 0.0
    else:
        fpr = float((negatives[prob_col] >= threshold).mean())
    metrics["fpr"] = fpr
    metrics["support"] = int(len(df))
    return metrics


def _compute_fairness_metrics(
    bundle,
    dataloader: DataLoader,
    transformer: nn.Module,
    lstm: nn.Module,
    device: torch.device,
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    records = _collect_sample_records(transformer, lstm, dataloader, device)
    enriched = records.merge(bundle.customers, on="customer_id", how="left")
    result: Dict[str, object] = {}
    for model_name, prob_col in [("transformer", "transformer_prob"), ("lstm", "lstm_prob")]:
        result[model_name] = {}
        for field in SEGMENT_FIELDS:
            result[model_name][field] = {}
            for value, group in enriched.groupby(field):
                result[model_name][field][str(value)] = _metric_dict_for_group(group, prob_col, thresholds[model_name])
    return result


def _early_warning_metrics(records: pd.DataFrame, customers: pd.DataFrame, prob_col: str, threshold: float) -> Dict[str, float]:
    customer_labels = customers[["customer_id", "customer_distress_label"]].copy()
    merged = records.merge(customer_labels, on="customer_id", how="left")
    distressed_records = merged[merged["days_to_first_distress"] >= 0]
    lead_times: List[float] = []
    detected_distressed = 0
    distressed_customer_ids = sorted(distressed_records["customer_id"].unique().tolist())
    for customer_id in distressed_customer_ids:
        customer_rows = distressed_records[distressed_records["customer_id"] == customer_id]
        positives = customer_rows[customer_rows[prob_col] >= threshold]
        if not positives.empty:
            detected_distressed += 1
            lead_times.append(float(positives["days_to_first_distress"].max()))

    stable_ids = customers[customers["customer_distress_label"] == 0]["customer_id"].unique().tolist()
    false_positives = 0
    for customer_id in stable_ids:
        customer_rows = merged[merged["customer_id"] == customer_id]
        if not customer_rows.empty and (customer_rows[prob_col] >= threshold).any():
            false_positives += 1

    return {
        "average_lead_time_days": round(float(np.mean(lead_times)) if lead_times else 0.0, 4),
        "hit_rate_on_distressed_customers": round(detected_distressed / max(1, len(distressed_customer_ids)), 4),
        "false_positive_rate_on_stable_customers": round(false_positives / max(1, len(stable_ids)), 4),
    }


def _event_mix_divergence(predicted_events: List[Dict[str, object]], actual_events: pd.DataFrame) -> float:
    predicted_counts = pd.Series([event["event_type"] for event in predicted_events]).value_counts(normalize=True)
    actual_counts = actual_events["event_type"].value_counts(normalize=True)
    all_events = sorted(set(predicted_counts.index).union(set(actual_counts.index)))
    divergence = 0.0
    for event_type in all_events:
        divergence += abs(float(predicted_counts.get(event_type, 0.0)) - float(actual_counts.get(event_type, 0.0)))
    return float(divergence / 2.0)


def _compute_simulation_metrics(
    bundle,
    transformer: nn.Module,
    lstm: nn.Module,
    device: torch.device,
    config: ProjectConfig,
    thresholds: Dict[str, float],
    test_dataloader: DataLoader,
) -> Dict[str, object]:
    test_frame = bundle.splits["test"].sort_values(["customer_id", "event_timestamp"]).reset_index(drop=True)
    test_customers = bundle.customers[bundle.customers["customer_id"].isin(test_frame["customer_id"].unique())].copy()
    sample_customer_ids = test_customers["customer_id"].head(20 if len(test_customers) >= 20 else len(test_customers)).tolist()

    event_divergences: List[float] = []
    balance_rmses: List[float] = []
    util_rmses: List[float] = []
    balance_stabilities: List[float] = []
    negative_count_stabilities: List[float] = []
    intervention_rows: List[Dict[str, object]] = []

    for customer_id in sample_customer_ids:
        customer_history = test_frame[test_frame["customer_id"] == customer_id].sort_values("event_timestamp").reset_index(drop=True)
        if len(customer_history) < 10:
            continue
        split_idx = max(5, int(len(customer_history) * 0.7))
        history = customer_history.iloc[:split_idx].copy()
        actual_continuation = customer_history.iloc[split_idx : split_idx + max(6, min(24, config.model.forecast_steps))].copy()
        if actual_continuation.empty:
            continue

        greedy = decode_forecast(
            transformer,
            history,
            intervention_type="none",
            horizon_days=30,
            device=device,
            context_length=config.model.context_length,
            strategy="greedy",
        )
        predicted_events = greedy["projected_events"][: len(actual_continuation)]
        predicted_balances = np.asarray([event["balance_after"] for event in predicted_events], dtype=float)
        predicted_utils = np.asarray([event["credit_utilization"] for event in predicted_events], dtype=float)
        actual_balances = actual_continuation["balance_after"].to_numpy(dtype=float)[: len(predicted_events)]
        actual_utils = actual_continuation["credit_utilization"].to_numpy(dtype=float)[: len(predicted_events)]
        if len(predicted_events) > 0:
            event_divergences.append(_event_mix_divergence(predicted_events, actual_continuation.iloc[: len(predicted_events)]))
            balance_rmses.append(float(np.sqrt(np.mean((predicted_balances - actual_balances) ** 2))))
            util_rmses.append(float(np.sqrt(np.mean((predicted_utils - actual_utils) ** 2))))

        sampled_runs = [
            decode_forecast(
                transformer,
                history,
                intervention_type="none",
                horizon_days=30,
                device=device,
                context_length=config.model.context_length,
                strategy="sample",
                temperature=0.8,
                top_k=3,
            )
            for _ in range(10)
        ]
        ending_balances = [run["balance_path"][-1] for run in sampled_runs]
        negative_counts = [run["forecast_summary"]["projected_negative_event_count"] for run in sampled_runs]
        balance_stabilities.append(float(np.std(ending_balances)))
        negative_count_stabilities.append(float(np.std(negative_counts)))

        baseline_generated = pd.DataFrame(greedy["projected_events"])
        baseline_history = pd.concat([history, baseline_generated], ignore_index=True, sort=False)
        baseline_risk = _lstm_probability_from_history(lstm, baseline_history, device, config.model.context_length)

        customer_meta = test_customers[test_customers["customer_id"] == customer_id].iloc[0].to_dict()
        for intervention in INTERVENTION_TYPES[1:]:
            adjusted_state = apply_intervention_policy(history_to_state(history), intervention)
            adjusted_history = apply_state_to_history(history, adjusted_state, intervention)
            intervention_forecast = decode_forecast(
                transformer,
                adjusted_history,
                intervention_type=intervention,
                horizon_days=30,
                device=device,
                context_length=config.model.context_length,
                strategy="greedy",
                starting_state=adjusted_state,
            )
            intervention_generated = pd.DataFrame(intervention_forecast["projected_events"])
            intervention_history = pd.concat([adjusted_history, intervention_generated], ignore_index=True, sort=False)
            intervention_risk = _lstm_probability_from_history(lstm, intervention_history, device, config.model.context_length)
            delta = baseline_risk - intervention_risk
            row = {
                "customer_id": customer_id,
                "intervention": intervention,
                "baseline_risk": baseline_risk,
                "intervention_risk": intervention_risk,
                "risk_reduction": delta,
                "material_improvement": int(delta >= 0.05),
            }
            for field in SEGMENT_FIELDS:
                row[field] = customer_meta.get(field)
            intervention_rows.append(row)

    intervention_df = pd.DataFrame(intervention_rows)
    per_intervention = {}
    portfolio_by_segment = {}
    for intervention, group in intervention_df.groupby("intervention"):
        per_intervention[intervention] = {
            "average_predicted_risk_reduction": round(float(group["risk_reduction"].mean()), 4),
            "share_improved_by_5pp": round(float(group["material_improvement"].mean()), 4),
            "customers_evaluated": int(len(group)),
        }
    for field in SEGMENT_FIELDS:
        portfolio_by_segment[field] = []
        if intervention_df.empty:
            continue
        grouped = intervention_df.groupby([field, "intervention"])
        for (segment_value, intervention), group in grouped:
            portfolio_by_segment[field].append(
                {
                    "segment_value": str(segment_value),
                    "intervention": str(intervention),
                    "customer_count": int(len(group)),
                    "average_baseline_risk": round(float(group["baseline_risk"].mean()), 4),
                    "average_intervention_risk": round(float(group["intervention_risk"].mean()), 4),
                    "average_risk_reduction": round(float(group["risk_reduction"].mean()), 4),
                }
            )

    sample_records = _collect_sample_records(transformer, lstm, test_dataloader, device)
    early_warning = {
        "transformer": _early_warning_metrics(sample_records, test_customers, "transformer_prob", thresholds["transformer"]),
        "lstm": _early_warning_metrics(sample_records, test_customers, "lstm_prob", thresholds["lstm"]),
    }

    return {
        "early_warning": early_warning,
        "simulation_quality": {
            "event_mix_divergence": round(float(np.mean(event_divergences)) if event_divergences else 0.0, 4),
            "balance_trajectory_rmse": round(float(np.mean(balance_rmses)) if balance_rmses else 0.0, 4),
            "utilization_rmse": round(float(np.mean(util_rmses)) if util_rmses else 0.0, 4),
        },
        "stability": {
            "ending_balance_std": round(float(np.mean(balance_stabilities)) if balance_stabilities else 0.0, 4),
            "negative_event_count_std": round(float(np.mean(negative_count_stabilities)) if negative_count_stabilities else 0.0, 4),
            "repeat_runs": 10,
        },
        "intervention_usefulness": {
            "per_intervention": per_intervention,
            "portfolio_by_segment": portfolio_by_segment,
        },
    }


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
            "due": len(DUE_BINS),
            "gap": len(GAP_BINS),
            "delta": len(DELTA_BINS),
            "intervention": len(bundle.encoders.intervention_type_to_id),
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

    ce_event = nn.CrossEntropyLoss()
    ce_bucket = nn.CrossEntropyLoss()
    pos_weight = _positive_class_weight(dataloaders["train"], device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    transformer_optimizer = torch.optim.Adam(
        transformer.parameters(),
        lr=getattr(config.model, "transformer_learning_rate", config.model.learning_rate),
    )
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
        f"transformer_lr={getattr(config.model, 'transformer_learning_rate', config.model.learning_rate)} "
        f"lstm_lr={config.model.learning_rate} "
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
            target_amount_bucket = batch["target_amount_bucket"].to(device)
            target_balance_delta_bucket = batch["target_balance_delta_bucket"].to(device)
            target_distress = batch["target_distress"].to(device)
            transformer_sample_weight = batch["transformer_sample_weight"].to(device)
            lstm_sample_weight = batch["lstm_sample_weight"].to(device)

            if not first_batch_logged:
                positive_rate = float(target_distress.mean().item())
                active_tokens = float(mask.sum(dim=1).float().mean().item())
                intervention_share = float((seq_tokens[:, :, 5].max(dim=1).values > 0).float().mean().item())
                print(
                    f"Epoch {epoch_idx + 1}/{config.model.max_epochs} | "
                    f"first_batch_size={target_distress.shape[0]} "
                    f"first_batch_positive_rate={positive_rate:.4f} "
                    f"avg_active_tokens={active_tokens:.1f} "
                    f"intervention_aug_share={intervention_share:.4f}"
                )
                first_batch_logged = True

            transformer_optimizer.zero_grad()
            transformer_outputs = transformer(seq_tokens, mask)
            transformer_distress_loss = (bce_loss(transformer_outputs["distress_logits"], target_distress) * transformer_sample_weight).sum() / transformer_sample_weight.sum().clamp(min=1.0)
            transformer_loss = (
                ce_event(transformer_outputs["next_event_logits"], target_event)
                + 0.5 * ce_bucket(transformer_outputs["next_amount_logits"], target_amount_bucket)
                + 0.5 * ce_bucket(transformer_outputs["next_balance_delta_logits"], target_balance_delta_bucket)
                + getattr(config.model, "distress_loss_weight", 1.0) * transformer_distress_loss
            )
            transformer_loss.backward()
            transformer_optimizer.step()
            transformer_loss_total += float(transformer_loss.item())

            if float(lstm_sample_weight.sum().item()) > 0:
                lstm_optimizer.zero_grad()
                lstm_logits = lstm(dense_features, mask)
                lstm_loss = (bce_loss(lstm_logits, target_distress) * lstm_sample_weight).sum() / lstm_sample_weight.sum().clamp(min=1.0)
                lstm_loss.backward()
                lstm_optimizer.step()
                lstm_loss_total += float(lstm_loss.item())
            batch_count += 1

        transformer_val_probs, transformer_val_targets = _collect_transformer_outputs(transformer, dataloaders["val"], device)
        lstm_val_probs, lstm_val_targets = _collect_lstm_outputs(lstm, dataloaders["val"], device)
        transformer_val = _compute_binary_metrics(transformer_val_targets, transformer_val_probs, threshold=0.5)
        lstm_val = _compute_binary_metrics(lstm_val_targets, lstm_val_probs, threshold=0.5)
        transformer_val_mean_prob = float(np.mean(transformer_val_probs)) if transformer_val_probs else 0.0
        transformer_val_positive_rate = float((np.asarray(transformer_val_probs) >= 0.5).mean()) if transformer_val_probs else 0.0

        transformer_improved = transformer_val["auc"] > best_transformer_auc
        if transformer_improved:
            best_transformer_auc = transformer_val["auc"]
            torch.save(transformer.state_dict(), transformer_path)
            patience_transformer = config.model.patience
        else:
            patience_transformer -= 1

        lstm_improved = lstm_val["auc"] > best_lstm_auc
        if lstm_improved:
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
            f"  Transformer val profile | mean_prob={transformer_val_mean_prob:.4f} "
            f"predicted_positive_rate@0.5={transformer_val_positive_rate:.4f}"
        )
        print(f"  Best so far | transformer_auc={best_transformer_auc:.4f} lstm_auc={best_lstm_auc:.4f}")

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
    transformer_selected_positive_rate = float((np.asarray(transformer_test_probs) >= transformer_threshold).mean()) if transformer_test_probs else 0.0

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
    thresholds = {"transformer": transformer_threshold, "lstm": lstm_threshold}
    fairness_metrics = _compute_fairness_metrics(bundle, dataloaders["test"], transformer, lstm, device, thresholds)
    simulation_metrics = _compute_simulation_metrics(bundle, transformer, lstm, device, config, thresholds, dataloaders["test"])

    print(f"Selected thresholds | transformer={transformer_threshold:.3f} lstm={lstm_threshold:.3f}")
    print(
        f"Transformer threshold profile | mean_val_prob={float(np.mean(transformer_val_probs)) if transformer_val_probs else 0.0:.4f} "
        f"predicted_positive_rate@selected={transformer_selected_positive_rate:.4f}"
    )
    print(f"Final transformer validation | {_format_metric_block(metrics['transformer']['validation'])}")
    print(f"Final transformer test       | {_format_metric_block(metrics['transformer']['test'])}")
    print(f"Final LSTM validation        | {_format_metric_block(metrics['lstm']['validation'])}")
    print(f"Final LSTM test              | {_format_metric_block(metrics['lstm']['test'])}")
    print(json.dumps(metrics, indent=2))

    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / "fairness_metrics.json").write_text(json.dumps(fairness_metrics, indent=2))
    (artifacts_dir / "simulation_metrics.json").write_text(json.dumps(simulation_metrics, indent=2))
    return TrainingArtifacts(metrics=metrics, model_paths={"transformer": transformer_path, "lstm": lstm_path})
