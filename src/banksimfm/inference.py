"""Public inference API for scoring, forecasting, and intervention simulation."""

from __future__ import annotations

from dataclasses import asdict
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
    GAP_BINS,
    UTIL_BINS,
    DemoBundle,
    build_encoders,
    load_or_create_demo_bundle,
)
from banksimfm.models.baseline import LSTMDistressModel
from banksimfm.models.transformer import CausalEventTransformer
from banksimfm.models.training import train_models
from banksimfm.runtime import resolve_device
from banksimfm.sim.engine import AccountState, apply_event, apply_intervention_policy, state_risk_factors
from banksimfm.types import ForecastResult, ScoreResult, SimulationResult, SimulationSide


INTERVENTIONS = [
    "reminder",
    "due_date_shift_7d",
    "temporary_overdraft_buffer",
    "installment_restructure",
]


def _coerce_history(history: pd.DataFrame | List[Dict[str, object]]) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        frame = history.copy()
    else:
        frame = pd.DataFrame(history)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    return frame.sort_values("event_timestamp").reset_index(drop=True)


def _bucketize_scalar(value: float, bins: np.ndarray) -> int:
    return int(np.digitize([float(value)], bins[1:], right=False)[0])


def _prepare_history_window(history: pd.DataFrame, context_length: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = history.copy()
    encoders = build_encoders()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    frame = frame.sort_values("event_timestamp").tail(context_length).reset_index(drop=True)
    frame["event_type_id"] = frame["event_type"].map(encoders.event_type_to_id).fillna(0).astype(int)
    frame["amount_bucket"] = frame["amount"].apply(lambda x: _bucketize_scalar(x, AMOUNT_BINS))
    frame["balance_bucket"] = frame["balance_after"].apply(lambda x: _bucketize_scalar(x, BALANCE_BINS))
    frame["util_bucket"] = frame["credit_utilization"].apply(lambda x: _bucketize_scalar(x, UTIL_BINS))
    frame["time_gap_days"] = frame["event_timestamp"].diff().dt.total_seconds().fillna(0) / 86400
    frame["time_gap_bucket"] = frame["time_gap_days"].apply(lambda x: _bucketize_scalar(x, GAP_BINS))
    frame["direction_flag"] = frame["amount_direction"].map({"credit": 1.0, "debit": -1.0}).fillna(0.0)
    frame["miss_flag"] = frame["event_type"].isin(["loan_emi_missed", "failed_debit"]).astype(float)
    frame["overdraft_flag"] = frame["event_type"].eq("overdraft_event").astype(float)

    seq_tokens = torch.zeros((1, context_length, 5), dtype=torch.long)
    dense_features = torch.zeros((1, context_length, 7), dtype=torch.float32)
    mask = torch.zeros((1, context_length), dtype=torch.bool)
    offset = context_length - len(frame)
    seq_tokens[0, offset:, :] = torch.tensor(
        frame[["event_type_id", "amount_bucket", "balance_bucket", "util_bucket", "time_gap_bucket"]].to_numpy(),
        dtype=torch.long,
    )
    dense_features[0, offset:, :] = torch.tensor(
        frame[
            [
                "amount",
                "balance_after",
                "credit_utilization",
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


def _history_to_state(history: pd.DataFrame) -> AccountState:
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


def _heuristic_probability(state: AccountState, horizon_days: int) -> float:
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
    if not transformer_path.exists() or not lstm_path.exists():
        return None

    device = resolve_device()
    transformer = CausalEventTransformer(
        vocab_size=len(build_encoders().event_type_to_id) + 1,
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
    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    transformer.eval()

    lstm = LSTMDistressModel(input_size=7, hidden_size=max(64, config.model.hidden_size // 2), dropout=config.model.dropout).to(device)
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm.eval()

    return {"device": device, "transformer": transformer, "lstm": lstm}


def _model_probability(history: pd.DataFrame) -> Optional[float]:
    loaded = _load_trained_models()
    if loaded is None or len(history) < 2:
        return None

    seq_tokens, dense_features, mask = _prepare_history_window(history)
    device = loaded["device"]
    seq_tokens = seq_tokens.to(device)
    dense_features = dense_features.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        _, transformer_logits = loaded["transformer"](seq_tokens, mask)
        lstm_logits = loaded["lstm"](dense_features, mask)
        transformer_prob = torch.sigmoid(transformer_logits).item()
        lstm_prob = torch.sigmoid(lstm_logits).item()
    return float((transformer_prob + lstm_prob) / 2.0)


def _top_drivers(state: AccountState) -> List[str]:
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
    state = _history_to_state(history_df)
    probability = _model_probability(history_df)
    if probability is None:
        probability = _heuristic_probability(state, horizon_days)
    else:
        probability = float(np.clip(probability + max(0.0, (horizon_days - 30) / 300), 0.01, 0.99))
    return ScoreResult(
        distress_probability=round(probability, 4),
        distress_label=probability >= 0.5,
        top_drivers=_top_drivers(state),
        recent_risk_signals=_recent_signals(history_df),
    )


def _forecast_steps_for_horizon(horizon_days: int) -> int:
    return max(6, min(24, horizon_days // 5))


def _choose_next_event(state: AccountState, rng: np.random.Generator) -> tuple[str, float, str]:
    if state.due_amount > 0 and state.due_in_days <= 2:
        if state.balance > state.due_amount * 0.9:
            return "loan_emi_paid", min(state.due_amount, max(50.0, state.balance * 0.15)), "loan"
        return "loan_emi_missed", max(50.0, state.due_amount), "loan"
    if state.balance < 250:
        return "failed_debit", max(35.0, state.due_amount or 80.0), "credit"
    if state.credit_utilization > 0.88:
        return "credit_card_payment_made", min(state.utilized_credit * 0.2, max(60.0, state.balance * 0.18)), "credit"
    draw = rng.random()
    if draw < 0.18:
        return "salary_credit", max(1800.0, state.last_income_amount or 2500.0), "income"
    if draw < 0.5:
        return "grocery_spend", float(rng.uniform(25, 120)), "daily_needs"
    if draw < 0.72:
        return "card_spend", float(rng.uniform(40, 180)), "credit"
    if draw < 0.86:
        return "utility_payment", float(rng.uniform(45, 110)), "utilities"
    return "transportation_spend", float(rng.uniform(10, 40)), "transport"


def forecast_customer(history: pd.DataFrame | List[Dict[str, object]], horizon_days: int = 30) -> ForecastResult:
    history_df = _coerce_history(history)
    state = _history_to_state(history_df)
    base_ts = pd.to_datetime(history_df["event_timestamp"].iloc[-1])
    rng = np.random.default_rng(abs(hash(str(history_df["customer_id"].iloc[-1]))) % (2 ** 32))
    projected_events: List[Dict[str, object]] = []
    balance_path = [round(state.balance, 2)]
    utilization_path = [round(state.credit_utilization, 4)]

    for step in range(_forecast_steps_for_horizon(horizon_days)):
        event_type, amount, category = _choose_next_event(state, rng)
        before_balance = state.balance
        state = apply_event(state, event_type, amount)
        event_ts = base_ts + pd.Timedelta(days=step + 1)
        projected_events.append(
            {
                "event_timestamp": event_ts.isoformat(),
                "event_type": event_type,
                "amount": round(amount, 2),
                "category": category,
                "balance_before": round(before_balance, 2),
                "balance_after": round(state.balance, 2),
                "credit_utilization": round(state.credit_utilization, 4),
                "intervention_flag": state.intervention_flag,
            }
        )
        balance_path.append(round(state.balance, 2))
        utilization_path.append(round(state.credit_utilization, 4))

    summary = {
        "forecast_horizon_days": horizon_days,
        "projected_distress_events": sum(1 for event in projected_events if event["event_type"] in {"loan_emi_missed", "failed_debit", "overdraft_event"}),
        "ending_balance": balance_path[-1],
        "ending_utilization": utilization_path[-1],
    }
    return ForecastResult(
        projected_events=projected_events,
        balance_path=balance_path,
        utilization_path=utilization_path,
        forecast_summary=summary,
    )


def simulate_intervention(
    history: pd.DataFrame | List[Dict[str, object]],
    intervention_type: str,
    horizon_days: int = 30,
) -> SimulationResult:
    if intervention_type not in INTERVENTIONS:
        raise ValueError(f"Unsupported intervention '{intervention_type}'.")

    history_df = _coerce_history(history)
    baseline_risk = score_customer(history_df, horizon_days=horizon_days)
    baseline_forecast = forecast_customer(history_df, horizon_days=horizon_days)

    state = _history_to_state(history_df)
    adjusted_state = apply_intervention_policy(state, intervention_type)
    last_row = history_df.iloc[-1].copy()
    last_row["balance_after"] = adjusted_state.balance
    last_row["credit_utilization"] = adjusted_state.credit_utilization
    last_row["days_to_next_due"] = adjusted_state.due_in_days
    last_row["intervention_flag"] = intervention_type
    adjusted_history = pd.concat([history_df.iloc[:-1], pd.DataFrame([last_row])], ignore_index=True)

    intervention_risk = score_customer(adjusted_history, horizon_days=horizon_days)
    intervention_forecast = forecast_customer(adjusted_history, horizon_days=horizon_days)

    scenario_differences = [
        f"Risk changes by {round(intervention_risk.distress_probability - baseline_risk.distress_probability, 4)} under {intervention_type}.",
        f"Ending balance moves from {baseline_forecast.balance_path[-1]:.2f} to {intervention_forecast.balance_path[-1]:.2f}.",
        f"Ending utilization moves from {baseline_forecast.utilization_path[-1]:.2f} to {intervention_forecast.utilization_path[-1]:.2f}.",
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
    metrics_path = config.artifacts_dir / "metrics.json"
    model_paths = [config.artifacts_dir / "transformer.pt", config.artifacts_dir / "lstm.pt"]
    if not metrics_path.exists() or not all(path.exists() for path in model_paths):
        train_models(config)
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
