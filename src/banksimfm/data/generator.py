"""Synthetic rule-based customer-event generation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from banksimfm.config import DataConfig
from banksimfm.data.schema import CustomerEvent
from banksimfm.sim.engine import AccountState, apply_event


ARCHETYPE_PARAMS: Dict[str, Dict[str, float]] = {
    "stable_salaried": {"income": 3200, "volatility": 0.08, "rent_ratio": 0.25, "spend_ratio": 0.22, "miss_prob": 0.02},
    "volatile_income": {"income": 2800, "volatility": 0.35, "rent_ratio": 0.24, "spend_ratio": 0.27, "miss_prob": 0.08},
    "high_obligation": {"income": 3000, "volatility": 0.12, "rent_ratio": 0.34, "spend_ratio": 0.21, "miss_prob": 0.09},
    "rising_utilization": {"income": 3100, "volatility": 0.15, "rent_ratio": 0.23, "spend_ratio": 0.3, "miss_prob": 0.1},
    "near_distress": {"income": 2500, "volatility": 0.2, "rent_ratio": 0.32, "spend_ratio": 0.31, "miss_prob": 0.18},
}


def _make_event(
    customer_id: str,
    timestamp: datetime,
    event_type: str,
    amount: float,
    direction: str,
    category: str,
    state_before: AccountState,
    state_after: AccountState,
    distress_label_30d: int = 0,
) -> CustomerEvent:
    return CustomerEvent(
        customer_id=customer_id,
        event_timestamp=timestamp,
        event_type=event_type,
        amount=round(float(amount), 2),
        amount_direction=direction,
        category=category,
        balance_before=round(state_before.balance, 2),
        balance_after=round(state_after.balance, 2),
        credit_limit=round(state_after.credit_limit, 2),
        credit_utilization=round(state_after.credit_utilization, 4),
        days_to_next_due=state_after.due_in_days,
        intervention_flag=state_after.intervention_flag,
        distress_label_30d=distress_label_30d,
    )


def _sample_amount(rng: np.random.Generator, base: float, volatility: float) -> float:
    value = rng.normal(loc=base, scale=max(1.0, base * volatility))
    return max(5.0, float(value))


def _generate_customer_events(
    customer_id: str,
    archetype: str,
    start_date: datetime,
    config: DataConfig,
    rng: np.random.Generator,
) -> List[CustomerEvent]:
    params = ARCHETYPE_PARAMS[archetype]
    state = AccountState(
        balance=float(rng.uniform(1200, 3500)),
        credit_limit=float(rng.choice([2500, 4000, 6000])),
        utilized_credit=float(rng.uniform(100, 900)),
    )
    events: List[CustomerEvent] = []

    for day in range(config.history_days):
        ts_day = start_date + timedelta(days=day)
        day_event_index = 0

        def next_timestamp() -> datetime:
            nonlocal day_event_index
            timestamp = ts_day.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(minutes=day_event_index * 7)
            day_event_index += 1
            return timestamp

        if ts_day.day == 1:
            income = _sample_amount(rng, params["income"], params["volatility"])
            before = replace(state)
            state = apply_event(state, "salary_credit", income)
            events.append(_make_event(customer_id, next_timestamp(), "salary_credit", income, "credit", "income", before, state))

        if ts_day.day in {2, 3}:
            rent = params["income"] * params["rent_ratio"] / 2
            before = replace(state)
            state = apply_event(state, "rent_payment", rent)
            events.append(_make_event(customer_id, next_timestamp(), "rent_payment", rent, "debit", "housing", before, state))

        if ts_day.day in {5, 18}:
            due_amount = params["income"] * 0.12
            before = replace(state)
            state = apply_event(state, "loan_emi_due", due_amount)
            events.append(_make_event(customer_id, next_timestamp(), "loan_emi_due", due_amount, "debit", "loan", before, state))

            if rng.random() < (1.0 - params["miss_prob"]) and state.balance > due_amount:
                before = replace(state)
                state = apply_event(state, "loan_emi_paid", due_amount)
                events.append(_make_event(customer_id, next_timestamp(), "loan_emi_paid", due_amount, "debit", "loan", before, state))
            else:
                before = replace(state)
                state = apply_event(state, "loan_emi_missed", due_amount)
                events.append(_make_event(customer_id, next_timestamp(), "loan_emi_missed", due_amount, "debit", "loan", before, state))

        if ts_day.day in {12, 26}:
            cc_due = params["income"] * 0.09
            before = replace(state)
            state = apply_event(state, "credit_card_payment_due", cc_due)
            events.append(_make_event(customer_id, next_timestamp(), "credit_card_payment_due", cc_due, "debit", "credit", before, state))

            if state.balance > cc_due and rng.random() > params["miss_prob"]:
                before = replace(state)
                state = apply_event(state, "credit_card_payment_made", cc_due)
                events.append(_make_event(customer_id, next_timestamp(), "credit_card_payment_made", cc_due, "debit", "credit", before, state))
            else:
                before = replace(state)
                state = apply_event(state, "failed_debit", cc_due)
                events.append(_make_event(customer_id, next_timestamp(), "failed_debit", cc_due, "debit", "credit", before, state))

        daily_spend_count = int(rng.integers(0, 3))
        for spend_idx in range(daily_spend_count):
            spend_base = params["income"] * params["spend_ratio"] / 12
            spend_amount = _sample_amount(rng, spend_base, 0.35)
            event_type = "card_spend" if rng.random() > 0.3 else "grocery_spend"
            category = "credit" if event_type == "card_spend" else "daily_needs"
            direction = "debit"
            before = replace(state)
            state = apply_event(state, event_type, spend_amount)
            events.append(_make_event(customer_id, next_timestamp(), event_type, spend_amount, direction, category, before, state))

        if rng.random() < 0.18:
            transport = _sample_amount(rng, 24, 0.25)
            before = replace(state)
            state = apply_event(state, "transportation_spend", transport)
            events.append(_make_event(customer_id, next_timestamp(), "transportation_spend", transport, "debit", "transport", before, state))

        if rng.random() < 0.12:
            utility = _sample_amount(rng, 65, 0.2)
            before = replace(state)
            state = apply_event(state, "utility_payment", utility)
            events.append(_make_event(customer_id, next_timestamp(), "utility_payment", utility, "debit", "utilities", before, state))

        if state.balance < 0 or state.credit_utilization > 0.9:
            before = replace(state)
            state = apply_event(state, "support_contact", 0.0)
            events.append(_make_event(customer_id, next_timestamp(), "support_contact", 0.0, "debit", "servicing", before, state))

        if state.balance < -50:
            before = replace(state)
            state = apply_event(state, "overdraft_event", abs(state.balance))
            events.append(_make_event(customer_id, next_timestamp(), "overdraft_event", abs(state.balance), "debit", "fees", before, state))

    return sorted(events, key=lambda item: item.event_timestamp)


def _apply_forward_distress_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["customer_id", "event_timestamp"]).reset_index(drop=True)
    distress_events = {"loan_emi_missed", "failed_debit", "overdraft_event"}
    timestamps = pd.to_datetime(frame["event_timestamp"])
    frame["distress_label_30d"] = 0

    for customer_id, group_idx in frame.groupby("customer_id").groups.items():
        indices = list(group_idx)
        group = frame.loc[indices]
        group_times = pd.to_datetime(group["event_timestamp"]).tolist()
        group_events = group["event_type"].tolist()

        for pos, row_idx in enumerate(indices):
            current_ts = group_times[pos]
            label = 0
            for future_pos in range(pos, len(indices)):
                if (group_times[future_pos] - current_ts).days > 30:
                    break
                if group_events[future_pos] in distress_events:
                    label = 1
                    break
            frame.at[row_idx, "distress_label_30d"] = label

    return frame


def generate_synthetic_dataset(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate event and customer summary datasets."""
    rng = np.random.default_rng(config.seed)
    start_date = datetime(2025, 1, 1)
    rows: List[Dict[str, object]] = []
    customers: List[Dict[str, object]] = []

    for idx in range(config.num_customers):
        archetype = config.archetypes[idx % len(config.archetypes)]
        customer_id = f"CUST_{idx:04d}"
        events = _generate_customer_events(customer_id, archetype, start_date, config, rng)
        rows.extend(event.to_dict() for event in events)
        customers.append({"customer_id": customer_id, "archetype": archetype})

    events_df = pd.DataFrame(rows)
    events_df = _apply_forward_distress_labels(events_df)
    customers_df = pd.DataFrame(customers)
    distress_rates = (
        events_df.groupby("customer_id")["distress_label_30d"].max().rename("customer_distress_label").reset_index()
    )
    customers_df = customers_df.merge(distress_rates, on="customer_id", how="left")
    return events_df, customers_df
