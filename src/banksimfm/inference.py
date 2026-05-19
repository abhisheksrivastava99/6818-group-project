"""Public inference API for scoring, forecasting, and intervention simulation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from banksimfm.config import ProjectConfig, default_config
from banksimfm.data.pipeline import (
    AMOUNT_BINS,
    BALANCE_BINS,
    DELTA_BINS,
    DUE_BINS,
    GAP_BINS,
    UTIL_BINS,
    DemoBundle,
    build_encoders,
    compute_due_amount_feature,
    load_or_create_demo_bundle,
)
from banksimfm.data.schema import INTERVENTION_TYPES
from banksimfm.models.baseline import LSTMDistressModel
from banksimfm.models.training import train_models
from banksimfm.models.transformer import CausalEventTransformer
from banksimfm.runtime import resolve_device
from banksimfm.sim.engine import apply_intervention_policy, state_risk_factors
from banksimfm.sim.scenario import apply_state_to_history, decode_forecast, history_to_state
from banksimfm.types import ForecastResult, ScoreResult, SimulationResult, SimulationSide


INTERVENTIONS = INTERVENTION_TYPES[1:]


def _coerce_history(history: pd.DataFrame | List[Dict[str, object]]) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        frame = history.copy()
    else:
        frame = pd.DataFrame(history)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    return frame.sort_values("event_timestamp").reset_index(drop=True)


def _bucketize_scalar(value: float, bins: np.ndarray) -> int:
    return int(np.digitize([float(value)], bins[1:], right=False)[0])


def _direction_flag(event_type: str, amount_direction: str | None = None) -> float:
    if amount_direction is not None:
        return {"credit": 1.0, "debit": -1.0}.get(amount_direction, 0.0)
    return 1.0 if event_type in {"salary_credit", "transfer_in"} else -1.0


def _prepare_history_window(
    history: pd.DataFrame,
    intervention_type: str = "none",
    context_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = history.copy()
    encoders = build_encoders()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    frame = frame.sort_values("event_timestamp").tail(context_length).reset_index(drop=True)
    frame["event_type_id"] = frame["event_type"].map(encoders.event_type_to_id).fillna(0).astype(int)
    frame["amount_bucket"] = frame["amount"].apply(lambda x: _bucketize_scalar(x, AMOUNT_BINS))
    frame["balance_bucket"] = frame["balance_after"].apply(lambda x: _bucketize_scalar(x, BALANCE_BINS))
    frame["util_bucket"] = frame["credit_utilization"].apply(lambda x: _bucketize_scalar(x, UTIL_BINS))
    if "due_amount_feature" not in frame.columns:
        frame["due_amount_feature"] = compute_due_amount_feature(frame)
    elif frame["due_amount_feature"].isna().any():
        frame["due_amount_feature"] = frame["due_amount_feature"].fillna(compute_due_amount_feature(frame))
    frame["due_bucket"] = frame["due_amount_feature"].apply(lambda x: _bucketize_scalar(x, DUE_BINS))
    frame["time_gap_days"] = frame["event_timestamp"].diff().dt.total_seconds().fillna(0) / 86400
    frame["time_gap_bucket"] = frame["time_gap_days"].apply(lambda x: _bucketize_scalar(x, GAP_BINS))
    frame["direction_flag"] = [
        _direction_flag(event_type, row.get("amount_direction"))
        for event_type, (_, row) in zip(frame["event_type"], frame.iterrows())
    ]
    frame["miss_flag"] = frame["event_type"].isin(["loan_emi_missed", "failed_debit"]).astype(float)
    frame["overdraft_flag"] = frame["event_type"].eq("overdraft_event").astype(float)
    frame["intervention_id"] = encoders.intervention_type_to_id[intervention_type]

    seq_tokens = torch.zeros((1, context_length, 7), dtype=torch.long)
    dense_features = torch.zeros((1, context_length, 8), dtype=torch.float32)
    mask = torch.zeros((1, context_length), dtype=torch.bool)
    offset = context_length - len(frame)
    seq_tokens[0, offset:, :] = torch.tensor(
        frame[["event_type_id", "amount_bucket", "balance_bucket", "util_bucket", "due_bucket", "time_gap_bucket", "intervention_id"]].to_numpy(),
        dtype=torch.long,
    )
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
    return seq_tokens, dense_features, mask


def _heuristic_probability(history_df: pd.DataFrame, horizon_days: int) -> float:
    state = history_to_state(history_df)
    factors = state_risk_factors(state)
    weighted = (
        0.28 * factors["low_balance"]
        + 0.24 * factors["high_utilization"]
        + 0.18 * factors["missed_payments"]
        + 0.12 * factors["failed_debits"]
        + 0.12 * factors["overdrafts"]
        + 0.06 * factors["income_instability"]
    )
    horizon_adj = min(0.18, max(0.0, (horizon_days - 30) / 300))
    return float(np.clip(weighted + horizon_adj, 0.01, 0.99))


@lru_cache(maxsize=1)
def _load_trained_models() -> Optional[Dict[str, object]]:
    config = default_config()
    transformer_path = config.artifacts_dir / "transformer.pt"
    lstm_path = config.artifacts_dir / "lstm.pt"
    metrics_path = config.artifacts_dir / "metrics.json"
    if not transformer_path.exists() or not lstm_path.exists():
        return None

    device = resolve_device()
    encoders = build_encoders()
    transformer = CausalEventTransformer(
        vocab_size=len(encoders.event_type_to_id) + 1,
        bucket_sizes={
            "amount": len(AMOUNT_BINS),
            "balance": len(BALANCE_BINS),
            "util": len(UTIL_BINS),
            "due": len(DUE_BINS),
            "gap": len(GAP_BINS),
            "delta": len(DELTA_BINS),
            "intervention": len(encoders.intervention_type_to_id),
        },
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    ).to(device)
    lstm = LSTMDistressModel(
        input_size=8,
        hidden_size=max(64, config.model.hidden_size // 2),
        dropout=config.model.dropout,
    ).to(device)
    try:
        transformer.load_state_dict(torch.load(transformer_path, map_location=device))
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    except RuntimeError:
        return None
    transformer.eval()
    lstm.eval()

    thresholds = {"transformer": 0.5, "lstm": 0.5}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        thresholds["transformer"] = float(metrics.get("transformer", {}).get("test", {}).get("threshold", 0.5))
        thresholds["lstm"] = float(metrics.get("lstm", {}).get("test", {}).get("threshold", 0.5))
    return {"device": device, "transformer": transformer, "lstm": lstm, "thresholds": thresholds}


def _lstm_probability(history: pd.DataFrame) -> Optional[float]:
    loaded = _load_trained_models()
    if loaded is None or len(history) < 2:
        return None
    _, dense_features, mask = _prepare_history_window(history)
    with torch.no_grad():
        logits = loaded["lstm"](dense_features.to(loaded["device"]), mask.to(loaded["device"]))
    return float(torch.sigmoid(logits).item())


def _transformer_probability(history: pd.DataFrame, intervention_type: str = "none") -> Optional[float]:
    loaded = _load_trained_models()
    if loaded is None or len(history) < 2:
        return None
    seq_tokens, _, mask = _prepare_history_window(history, intervention_type=intervention_type)
    with torch.no_grad():
        outputs = loaded["transformer"](seq_tokens.to(loaded["device"]), mask.to(loaded["device"]))
    return float(torch.sigmoid(outputs["distress_logits"]).item())


def _top_drivers(history: pd.DataFrame) -> List[str]:
    state = history_to_state(history)
    factors = state_risk_factors(state)
    label_map = {
        "low_balance": "recent low balances",
        "high_utilization": "rising credit utilization",
        "missed_payments": "missed due events",
        "failed_debits": "repeated debit failures",
        "overdrafts": "overdraft activity",
        "income_instability": "income instability",
    }
    ranked = sorted(factors.items(), key=lambda item: item[1], reverse=True)
    return [label_map[name] for name, score in ranked if score > 0.15][:3]


def _recent_signals(history: pd.DataFrame) -> List[str]:
    signals: List[str] = []
    recent = history.tail(12)
    if (recent["balance_after"] < 200).any():
        signals.append("Low-balance events detected in the recent history.")
    if recent["event_type"].isin(["loan_emi_missed", "failed_debit"]).any():
        signals.append("The customer has recent payment friction or failed debits.")
    if recent["credit_utilization"].max() > 0.85:
        signals.append("Credit utilization spiked above 85% recently.")
    if not signals:
        signals.append("Recent activity is comparatively stable.")
    return signals


def score_customer(history: pd.DataFrame | List[Dict[str, object]], horizon_days: int = 30) -> ScoreResult:
    history_df = _coerce_history(history)
    probability = _lstm_probability(history_df)
    loaded = _load_trained_models()
    threshold = 0.5 if loaded is None else float(loaded["thresholds"]["lstm"])
    if probability is None:
        probability = _heuristic_probability(history_df, horizon_days)
    else:
        probability = float(np.clip(probability + max(0.0, (horizon_days - 30) / 300), 0.01, 0.99))
    return ScoreResult(
        distress_probability=round(probability, 4),
        distress_label=probability >= threshold,
        top_drivers=_top_drivers(history_df),
        recent_risk_signals=_recent_signals(history_df),
    )


def _heuristic_forecast(history_df: pd.DataFrame, horizon_days: int) -> Dict[str, object]:
    state = history_to_state(history_df)
    base_ts = pd.to_datetime(history_df["event_timestamp"].iloc[-1])
    projected_events: List[Dict[str, object]] = []
    balance_path = [round(state.balance, 2)]
    utilization_path = [round(state.credit_utilization, 4)]
    for step in range(max(6, min(24, horizon_days // 5))):
        event_type = "grocery_spend" if state.balance > 200 else "failed_debit"
        amount = 75.0 if event_type == "grocery_spend" else 45.0
        before_balance = state.balance
        from banksimfm.sim.engine import apply_event  # local import to avoid wider module noise

        state = apply_event(state, event_type, amount)
        projected_events.append(
            {
                "customer_id": history_df.iloc[-1]["customer_id"],
                "event_timestamp": (base_ts + pd.Timedelta(days=step + 1)).isoformat(),
                "event_type": event_type,
                "amount": amount,
                "amount_direction": "debit",
                "category": event_type.replace("_", " "),
                "balance_before": round(before_balance, 2),
                "balance_after": round(state.balance, 2),
                "credit_limit": float(history_df.iloc[-1].get("credit_limit", 0.0)),
                "credit_utilization": round(state.credit_utilization, 4),
                "due_amount_feature": round(state.due_amount, 2),
                "days_to_next_due": state.due_in_days,
                "intervention_flag": "none",
            }
        )
        balance_path.append(round(state.balance, 2))
        utilization_path.append(round(state.credit_utilization, 4))
    return {
        "projected_events": projected_events,
        "balance_path": balance_path,
        "utilization_path": utilization_path,
        "forecast_summary": {
            "forecast_horizon_days": horizon_days,
            "generation_mode": "heuristic_fallback",
            "top_negative_events": [],
            "projected_negative_event_count": 0,
            "path_level_risk_summary": {"average_predicted_distress_probability": 0.0, "max_predicted_distress_probability": 0.0},
            "scenario_metrics": {"ending_balance": balance_path[-1], "ending_utilization": utilization_path[-1], "negative_event_share": 0.0},
        },
    }


def forecast_customer(history: pd.DataFrame | List[Dict[str, object]], horizon_days: int = 30) -> ForecastResult:
    history_df = _coerce_history(history)
    loaded = _load_trained_models()
    if loaded is None:
        decoded = _heuristic_forecast(history_df, horizon_days)
    else:
        decoded = decode_forecast(
            loaded["transformer"],
            history_df,
            intervention_type="none",
            horizon_days=horizon_days,
            device=loaded["device"],
            context_length=default_config().model.context_length,
            strategy="greedy",
        )
    return ForecastResult(
        projected_events=decoded["projected_events"],
        balance_path=decoded["balance_path"],
        utilization_path=decoded["utilization_path"],
        forecast_summary=decoded["forecast_summary"],
    )


def simulate_intervention(
    history: pd.DataFrame | List[Dict[str, object]],
    intervention_type: str,
    horizon_days: int = 30,
) -> SimulationResult:
    if intervention_type not in INTERVENTIONS:
        raise ValueError(f"Unsupported intervention '{intervention_type}'.")

    history_df = _coerce_history(history)
    loaded = _load_trained_models()
    baseline_forecast = forecast_customer(history_df, horizon_days=horizon_days)

    state = history_to_state(history_df)
    adjusted_state = apply_intervention_policy(state, intervention_type)
    adjusted_history = apply_state_to_history(history_df, adjusted_state, intervention_type)

    if loaded is None:
        intervention_decoded = _heuristic_forecast(adjusted_history, horizon_days)
    else:
        intervention_decoded = decode_forecast(
            loaded["transformer"],
            adjusted_history,
            intervention_type=intervention_type,
            horizon_days=horizon_days,
            device=loaded["device"],
            context_length=default_config().model.context_length,
            strategy="greedy",
            starting_state=adjusted_state,
        )
    intervention_forecast = ForecastResult(
        projected_events=intervention_decoded["projected_events"],
        balance_path=intervention_decoded["balance_path"],
        utilization_path=intervention_decoded["utilization_path"],
        forecast_summary=intervention_decoded["forecast_summary"],
    )

    baseline_history = pd.concat([history_df, pd.DataFrame(baseline_forecast.projected_events)], ignore_index=True, sort=False)
    intervention_history = pd.concat([adjusted_history, pd.DataFrame(intervention_forecast.projected_events)], ignore_index=True, sort=False)
    baseline_risk = score_customer(baseline_history, horizon_days=horizon_days)
    intervention_risk = score_customer(intervention_history, horizon_days=horizon_days)

    scenario_differences = [
        f"Risk changes by {round(intervention_risk.distress_probability - baseline_risk.distress_probability, 4)} under {intervention_type}.",
        f"Ending balance moves from {baseline_forecast.balance_path[-1]:.2f} to {intervention_forecast.balance_path[-1]:.2f}.",
        f"Ending utilization moves from {baseline_forecast.utilization_path[-1]:.2f} to {intervention_forecast.utilization_path[-1]:.2f}.",
        f"Projected negative events move from {baseline_forecast.forecast_summary['projected_negative_event_count']} to {intervention_forecast.forecast_summary['projected_negative_event_count']}.",
    ]
    return SimulationResult(
        baseline=SimulationSide(risk=baseline_risk, forecast=baseline_forecast),
        intervention=SimulationSide(risk=intervention_risk, forecast=intervention_forecast),
        risk_delta=round(intervention_risk.distress_probability - baseline_risk.distress_probability, 4),
        scenario_differences=scenario_differences,
    )


def ensure_demo_artifacts(config: ProjectConfig | None = None) -> DemoBundle:
    config = config or default_config()
    bundle = load_or_create_demo_bundle(config)
    required_artifacts = [
        config.artifacts_dir / "metrics.json",
        config.artifacts_dir / "fairness_metrics.json",
        config.artifacts_dir / "simulation_metrics.json",
        config.artifacts_dir / "transformer.pt",
        config.artifacts_dir / "lstm.pt",
    ]
    if not all(path.exists() for path in required_artifacts):
        train_models(config)
        _load_trained_models.cache_clear()
    elif _load_trained_models() is None:
        train_models(config)
        _load_trained_models.cache_clear()
    return bundle


def get_representative_customers(bundle: Optional[DemoBundle] = None) -> pd.DataFrame:
    bundle = bundle or ensure_demo_artifacts()
    merged = bundle.customers.merge(
        bundle.events.groupby("customer_id")["distress_label_30d"].max().rename("distress_flag"),
        on="customer_id",
        how="left",
    )
    picks = (
        merged.sort_values(["archetype", "distress_flag"], ascending=[True, False])
        .groupby("archetype")
        .head(2)
        .reset_index(drop=True)
    )
    return picks
