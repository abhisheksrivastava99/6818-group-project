"""Shared transformer decoding and scenario helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from banksimfm.data.pipeline import (
    AMOUNT_BINS,
    BALANCE_BINS,
    DELTA_BINS,
    DUE_BINS,
    GAP_BINS,
    UTIL_BINS,
    DISTRESS_EVENT_TYPES,
    build_encoders,
    compute_due_amount_feature,
)
from banksimfm.data.schema import INTERVENTION_TYPES
from banksimfm.sim.engine import AccountState, apply_event


NEGATIVE_EVENT_TYPES = DISTRESS_EVENT_TYPES


def _bucketize_scalar(value: float, bins: np.ndarray) -> int:
    return int(np.digitize([float(value)], bins[1:], right=False)[0])


def bucket_center(bucket_id: int, bins: np.ndarray) -> float:
    bucket_id = int(np.clip(bucket_id, 0, len(bins) - 2))
    lower = bins[bucket_id]
    upper = bins[bucket_id + 1]
    return float((lower + upper) / 2.0)


def history_to_state(history: pd.DataFrame) -> AccountState:
    last_row = history.iloc[-1]
    missed = int(history["event_type"].isin(["loan_emi_missed"]).sum())
    failed_debits = int(history["event_type"].eq("failed_debit").sum())
    overdrafts = int(history["event_type"].eq("overdraft_event").sum())
    low_balance_streak = int((history["balance_after"] < 200).tail(5).sum())
    due_amount = float(last_row.get("due_amount_feature", np.nan))
    if np.isnan(due_amount):
        due_amount = float(compute_due_amount_feature(history).iloc[-1]) if len(history) else 0.0
    return AccountState(
        balance=float(last_row["balance_after"]),
        credit_limit=float(last_row["credit_limit"]),
        utilized_credit=float(last_row["credit_utilization"]) * float(last_row["credit_limit"]),
        missed_payments=missed,
        overdraft_events=overdrafts,
        failed_debits=failed_debits,
        low_balance_streak=low_balance_streak,
        due_amount=due_amount,
        due_in_days=int(last_row.get("days_to_next_due", 7) or 7),
        last_income_amount=float(history.loc[history["event_type"] == "salary_credit", "amount"].tail(1).iloc[0] if not history.loc[history["event_type"] == "salary_credit"].empty else 0.0),
        intervention_flag=str(last_row.get("intervention_flag", "none")),
    )


def apply_state_to_history(history: pd.DataFrame, state: AccountState, intervention_type: str) -> pd.DataFrame:
    adjusted = history.copy()
    if "due_amount_feature" not in adjusted.columns:
        adjusted["due_amount_feature"] = compute_due_amount_feature(adjusted)
    elif adjusted["due_amount_feature"].isna().any():
        adjusted["due_amount_feature"] = adjusted["due_amount_feature"].fillna(compute_due_amount_feature(adjusted))
    last_idx = adjusted.index[-1]
    adjusted.loc[last_idx, "balance_after"] = state.balance
    adjusted.loc[last_idx, "credit_utilization"] = state.credit_utilization
    adjusted.loc[last_idx, "days_to_next_due"] = state.due_in_days
    adjusted.loc[last_idx, "intervention_flag"] = intervention_type
    adjusted.loc[last_idx, "due_amount_feature"] = state.due_amount
    return adjusted.reset_index(drop=True)


def _prepare_transformer_window(history: pd.DataFrame, intervention_type: str, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoders = build_encoders()
    frame = history.copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    frame = frame.sort_values("event_timestamp").tail(context_length).reset_index(drop=True)
    if "due_amount_feature" not in frame.columns:
        frame["due_amount_feature"] = compute_due_amount_feature(frame)
    elif frame["due_amount_feature"].isna().any():
        frame["due_amount_feature"] = frame["due_amount_feature"].fillna(compute_due_amount_feature(frame))
    frame["event_type_id"] = frame["event_type"].map(encoders.event_type_to_id).fillna(0).astype(int)
    frame["amount_bucket"] = frame["amount"].apply(lambda x: _bucketize_scalar(x, AMOUNT_BINS))
    frame["balance_bucket"] = frame["balance_after"].apply(lambda x: _bucketize_scalar(x, BALANCE_BINS))
    frame["util_bucket"] = frame["credit_utilization"].apply(lambda x: _bucketize_scalar(x, UTIL_BINS))
    frame["due_bucket"] = frame["due_amount_feature"].apply(lambda x: _bucketize_scalar(x, DUE_BINS))
    frame["time_gap_days"] = frame["event_timestamp"].diff().dt.total_seconds().fillna(0) / 86400
    frame["time_gap_bucket"] = frame["time_gap_days"].apply(lambda x: _bucketize_scalar(x, GAP_BINS))
    frame["intervention_id"] = encoders.intervention_type_to_id[intervention_type]

    seq_tokens = torch.zeros((1, context_length, 7), dtype=torch.long)
    mask = torch.zeros((1, context_length), dtype=torch.bool)
    offset = context_length - len(frame)
    seq_tokens[0, offset:, :] = torch.tensor(
        frame[["event_type_id", "amount_bucket", "balance_bucket", "util_bucket", "due_bucket", "time_gap_bucket", "intervention_id"]].to_numpy(),
        dtype=torch.long,
    )
    mask[0, offset:] = True
    return seq_tokens, mask


def _sample_from_logits(logits: torch.Tensor, strategy: str, temperature: float, top_k: int) -> int:
    if strategy == "greedy":
        return int(torch.argmax(logits).item())
    scaled = logits / max(temperature, 1e-6)
    values, indices = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
    probs = torch.softmax(values, dim=-1)
    picked = int(torch.multinomial(probs, num_samples=1).item())
    return int(indices[picked].item())


def _decode_amount(event_type: str, raw_amount: float, state: AccountState) -> float:
    if event_type == "support_contact":
        return 0.0
    if event_type == "overdraft_event":
        return max(abs(state.balance), max(15.0, raw_amount))
    if event_type == "credit_card_payment_made":
        return min(max(30.0, raw_amount), max(30.0, state.utilized_credit))
    if event_type == "loan_emi_paid":
        return min(max(40.0, raw_amount), max(40.0, state.due_amount or raw_amount))
    if event_type in {"salary_credit", "transfer_in"}:
        return max(50.0, raw_amount)
    return max(5.0, abs(raw_amount))


def decode_forecast(
    transformer: torch.nn.Module,
    history: pd.DataFrame,
    intervention_type: str,
    horizon_days: int,
    device: torch.device,
    context_length: int = 256,
    strategy: str = "greedy",
    temperature: float = 0.8,
    top_k: int = 3,
    starting_state: Optional[AccountState] = None,
) -> Dict[str, Any]:
    encoders = build_encoders()
    history_df = history.copy()
    history_df["event_timestamp"] = pd.to_datetime(history_df["event_timestamp"])
    history_df = history_df.sort_values("event_timestamp").reset_index(drop=True)
    if "due_amount_feature" not in history_df.columns:
        history_df["due_amount_feature"] = compute_due_amount_feature(history_df)
    elif history_df["due_amount_feature"].isna().any():
        history_df["due_amount_feature"] = history_df["due_amount_feature"].fillna(compute_due_amount_feature(history_df))
    state = starting_state if starting_state is not None else history_to_state(history_df)
    base_ts = pd.to_datetime(history_df["event_timestamp"].iloc[-1])
    projected_events: List[Dict[str, Any]] = []
    balance_path = [round(state.balance, 2)]
    utilization_path = [round(state.credit_utilization, 4)]
    distress_probs: List[float] = []
    predicted_negative_events: List[str] = []
    steps = max(6, min(24, horizon_days // 5))
    working_history = history_df.copy()

    for step in range(steps):
        seq_tokens, mask = _prepare_transformer_window(working_history, intervention_type, context_length)
        with torch.no_grad():
            outputs = transformer(seq_tokens.to(device), mask.to(device))
        next_event_id = _sample_from_logits(outputs["next_event_logits"][0], strategy, temperature, top_k)
        next_amount_id = _sample_from_logits(outputs["next_amount_logits"][0], strategy, temperature, top_k)
        next_delta_id = _sample_from_logits(outputs["next_balance_delta_logits"][0], strategy, temperature, top_k)

        event_type = encoders.event_id_to_type.get(next_event_id, "grocery_spend")
        raw_amount = bucket_center(next_amount_id, AMOUNT_BINS)
        amount = _decode_amount(event_type, raw_amount, state)
        before_balance = state.balance
        state = apply_event(state, event_type, amount)
        event_ts = base_ts + pd.Timedelta(days=step + 1)
        distress_prob = float(torch.sigmoid(outputs["distress_logits"]).item())
        distress_probs.append(distress_prob)
        if event_type in NEGATIVE_EVENT_TYPES:
            predicted_negative_events.append(event_type)

        projected_event = {
            "customer_id": working_history.iloc[-1]["customer_id"],
            "event_timestamp": event_ts.isoformat(),
            "event_type": event_type,
            "amount": round(amount, 2),
            "amount_direction": "credit" if event_type in {"salary_credit", "transfer_in"} else "debit",
            "category": event_type.replace("_", " "),
            "balance_before": round(before_balance, 2),
            "balance_after": round(state.balance, 2),
            "credit_limit": float(working_history.iloc[-1].get("credit_limit", state.credit_limit)),
            "credit_utilization": round(state.credit_utilization, 4),
            "due_amount_feature": round(state.due_amount, 2),
            "days_to_next_due": int(state.due_in_days),
            "intervention_flag": intervention_type,
            "predicted_balance_delta": round(bucket_center(next_delta_id, DELTA_BINS), 2),
            "predicted_distress_probability": round(distress_prob, 4),
        }
        projected_events.append(projected_event)
        balance_path.append(round(state.balance, 2))
        utilization_path.append(round(state.credit_utilization, 4))

        history_row = {
            "customer_id": working_history.iloc[-1]["customer_id"],
            "event_timestamp": event_ts,
            "event_type": event_type,
            "amount": amount,
            "amount_direction": "credit" if event_type in {"salary_credit", "transfer_in"} else "debit",
            "category": projected_event["category"],
            "balance_before": before_balance,
            "balance_after": state.balance,
            "credit_limit": working_history.iloc[-1]["credit_limit"],
            "credit_utilization": state.credit_utilization,
            "due_amount_feature": state.due_amount,
            "days_to_next_due": state.due_in_days,
            "intervention_flag": intervention_type,
            "distress_label_30d": int(distress_prob >= 0.5),
        }
        working_history = pd.concat([working_history, pd.DataFrame([history_row])], ignore_index=True)

    counter = Counter(predicted_negative_events)
    top_negative_events = [
        {"event_type": event_type, "count": count}
        for event_type, count in counter.most_common(3)
    ]
    return {
        "projected_events": projected_events,
        "balance_path": balance_path,
        "utilization_path": utilization_path,
        "forecast_summary": {
            "forecast_horizon_days": horizon_days,
            "generation_mode": "transformer_greedy" if strategy == "greedy" else "transformer_sampled",
            "top_negative_events": top_negative_events,
            "projected_negative_event_count": len(predicted_negative_events),
            "path_level_risk_summary": {
                "average_predicted_distress_probability": round(float(np.mean(distress_probs)) if distress_probs else 0.0, 4),
                "max_predicted_distress_probability": round(float(np.max(distress_probs)) if distress_probs else 0.0, 4),
            },
            "scenario_metrics": {
                "ending_balance": balance_path[-1],
                "ending_utilization": utilization_path[-1],
                "negative_event_share": round(len(predicted_negative_events) / max(1, len(projected_events)), 4),
            },
        },
    }
