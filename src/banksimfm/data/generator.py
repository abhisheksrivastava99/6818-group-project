"""Synthetic rule-based customer-event generation with softer archetypes."""

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
    "stable_salaried": {
        "income_min": 2900,
        "income_max": 3800,
        "volatility_min": 0.05,
        "volatility_max": 0.12,
        "rent_ratio_min": 0.21,
        "rent_ratio_max": 0.29,
        "spend_ratio_min": 0.18,
        "spend_ratio_max": 0.24,
        "miss_tendency_min": 0.0,
        "miss_tendency_max": 0.04,
        "recovery_min": 0.65,
        "recovery_max": 0.9,
        "buffer_min": 0.92,
        "buffer_max": 1.32,
        "credit_use_min": 0.05,
        "credit_use_max": 0.18,
    },
    "volatile_income": {
        "income_min": 2200,
        "income_max": 3400,
        "volatility_min": 0.22,
        "volatility_max": 0.45,
        "rent_ratio_min": 0.22,
        "rent_ratio_max": 0.3,
        "spend_ratio_min": 0.22,
        "spend_ratio_max": 0.31,
        "miss_tendency_min": 0.03,
        "miss_tendency_max": 0.1,
        "recovery_min": 0.35,
        "recovery_max": 0.7,
        "buffer_min": 0.55,
        "buffer_max": 1.1,
        "credit_use_min": 0.1,
        "credit_use_max": 0.26,
    },
    "high_obligation": {
        "income_min": 2600,
        "income_max": 3400,
        "volatility_min": 0.08,
        "volatility_max": 0.18,
        "rent_ratio_min": 0.31,
        "rent_ratio_max": 0.4,
        "spend_ratio_min": 0.18,
        "spend_ratio_max": 0.25,
        "miss_tendency_min": 0.06,
        "miss_tendency_max": 0.15,
        "recovery_min": 0.28,
        "recovery_max": 0.58,
        "buffer_min": 0.42,
        "buffer_max": 0.9,
        "credit_use_min": 0.14,
        "credit_use_max": 0.3,
    },
    "rising_utilization": {
        "income_min": 2700,
        "income_max": 3500,
        "volatility_min": 0.1,
        "volatility_max": 0.2,
        "rent_ratio_min": 0.2,
        "rent_ratio_max": 0.28,
        "spend_ratio_min": 0.26,
        "spend_ratio_max": 0.35,
        "miss_tendency_min": 0.04,
        "miss_tendency_max": 0.1,
        "recovery_min": 0.3,
        "recovery_max": 0.6,
        "buffer_min": 0.55,
        "buffer_max": 1.05,
        "credit_use_min": 0.22,
        "credit_use_max": 0.48,
    },
    "near_distress": {
        "income_min": 2200,
        "income_max": 3000,
        "volatility_min": 0.15,
        "volatility_max": 0.28,
        "rent_ratio_min": 0.28,
        "rent_ratio_max": 0.37,
        "spend_ratio_min": 0.27,
        "spend_ratio_max": 0.33,
        "miss_tendency_min": 0.06,
        "miss_tendency_max": 0.14,
        "recovery_min": 0.3,
        "recovery_max": 0.65,
        "buffer_min": 0.42,
        "buffer_max": 0.92,
        "credit_use_min": 0.28,
        "credit_use_max": 0.58,
    },
}


def _sample_profile(archetype: str, rng: np.random.Generator) -> Dict[str, float]:
    params = ARCHETYPE_PARAMS[archetype]
    income = float(rng.uniform(params["income_min"], params["income_max"]))
    profile = {
        "income": income,
        "volatility": float(rng.uniform(params["volatility_min"], params["volatility_max"])),
        "rent_ratio": float(rng.uniform(params["rent_ratio_min"], params["rent_ratio_max"])),
        "spend_ratio": float(rng.uniform(params["spend_ratio_min"], params["spend_ratio_max"])),
        "miss_tendency": float(rng.uniform(params["miss_tendency_min"], params["miss_tendency_max"])),
        "recovery_tendency": float(rng.uniform(params["recovery_min"], params["recovery_max"])),
        "buffer_multiplier": float(rng.uniform(params["buffer_min"], params["buffer_max"])),
        "credit_use_ratio": float(rng.uniform(params["credit_use_min"], params["credit_use_max"])),
        "salary_day": int(rng.integers(1, 4)),
        "rent_day": int(rng.integers(2, 6)),
        "loan_due_day": int(rng.integers(7, 21)),
        "credit_due_day": int(rng.integers(14, 27)),
    }
    profile["income_band"] = _income_band(profile["income"])
    profile["employment_type"] = _employment_type(archetype, rng)
    profile["region"] = _region(rng)
    profile["risk_segment"] = _risk_segment(archetype, profile)
    return profile


def _income_band(income: float) -> str:
    if income < 2500:
        return "low"
    if income < 3300:
        return "middle"
    return "upper_middle"


def _employment_type(archetype: str, rng: np.random.Generator) -> str:
    if archetype == "stable_salaried":
        choices = ["full_time", "professional", "government"]
        probs = [0.65, 0.2, 0.15]
    elif archetype == "volatile_income":
        choices = ["gig", "contract", "self_employed"]
        probs = [0.45, 0.25, 0.3]
    elif archetype == "high_obligation":
        choices = ["full_time", "service", "contract"]
        probs = [0.55, 0.25, 0.2]
    elif archetype == "rising_utilization":
        choices = ["full_time", "service", "small_business"]
        probs = [0.45, 0.35, 0.2]
    else:
        choices = ["service", "contract", "gig"]
        probs = [0.35, 0.3, 0.35]
    return str(rng.choice(choices, p=probs))


def _region(rng: np.random.Generator) -> str:
    return str(rng.choice(["north", "south", "east", "west", "central"], p=[0.18, 0.18, 0.21, 0.19, 0.24]))


def _risk_segment(archetype: str, profile: Dict[str, float]) -> str:
    if archetype == "near_distress":
        return "elevated"
    if archetype == "rising_utilization":
        return "watchlist"
    if archetype == "high_obligation":
        return "burdened"
    if archetype == "volatile_income":
        return "income_volatility"
    if profile["miss_tendency"] > 0.03:
        return "emerging_risk"
    return "stable"


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


def _monthly_day(timestamp: datetime) -> int:
    return ((timestamp - datetime(timestamp.year, timestamp.month, 1)).days) + 1


def _liquidity_stress(state: AccountState, profile: Dict[str, float]) -> float:
    balance_pressure = max(0.0, (800.0 - state.balance) / 800.0)
    utilization_pressure = max(0.0, state.credit_utilization - 0.5)
    missed_pressure = min(1.0, state.missed_payments / 3.0)
    return float(
        np.clip(
            0.4 * balance_pressure
            + 0.25 * utilization_pressure
            + 0.2 * missed_pressure
            + 0.15 * profile["volatility"],
            0.0,
            1.0,
        )
    )


def _payment_success_probability(
    state: AccountState,
    profile: Dict[str, float],
    due_amount: float,
    monthly_shock: float,
) -> float:
    coverage = np.clip((state.balance + 0.35 * max(0.0, state.last_income_amount)) / max(1.0, due_amount), 0.0, 1.5)
    stress = _liquidity_stress(state, profile)
    probability = (
        0.82 * coverage
        + 0.28 * profile["recovery_tendency"]
        - 0.45 * profile["miss_tendency"]
        - 0.3 * stress
        - 0.12 * monthly_shock
    )
    return float(np.clip(probability, 0.08, 0.985))


def _should_emit_support_contact(state: AccountState, profile: Dict[str, float], rng: np.random.Generator) -> bool:
    stress = _liquidity_stress(state, profile)
    support_prob = 0.04 + 0.22 * stress + 0.08 * (state.credit_utilization > 0.88)
    return bool(rng.random() < np.clip(support_prob, 0.0, 0.55))


def _generate_customer_events(
    customer_id: str,
    archetype: str,
    start_date: datetime,
    config: DataConfig,
    rng: np.random.Generator,
) -> Tuple[List[CustomerEvent], Dict[str, object]]:
    profile = _sample_profile(archetype, rng)
    credit_limit = float(rng.choice([2500, 4000, 6000]))
    state = AccountState(
        balance=float(rng.uniform(1600, 3800) * profile["buffer_multiplier"]),
        credit_limit=credit_limit,
        utilized_credit=float(credit_limit * profile["credit_use_ratio"]),
    )
    events: List[CustomerEvent] = []

    for day in range(config.history_days):
        ts_day = start_date + timedelta(days=day)
        day_event_index = 0
        month_day = _monthly_day(ts_day)
        monthly_shock = float(np.clip(rng.normal(0.0, profile["volatility"] * 0.6), -0.22, 0.28))
        spending_pressure = float(np.clip(profile["spend_ratio"] + monthly_shock * 0.15, 0.14, 0.42))

        def next_timestamp() -> datetime:
            nonlocal day_event_index
            timestamp = ts_day.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(minutes=day_event_index * 7)
            day_event_index += 1
            return timestamp

        if month_day == profile["salary_day"]:
            income = _sample_amount(rng, profile["income"] * (1.0 - 0.2 * max(0.0, monthly_shock)), profile["volatility"])
            before = replace(state)
            state = apply_event(state, "salary_credit", income)
            events.append(_make_event(customer_id, next_timestamp(), "salary_credit", income, "credit", "income", before, state))

        if month_day in {profile["rent_day"], min(28, profile["rent_day"] + 1)}:
            rent = profile["income"] * profile["rent_ratio"] / 2
            before = replace(state)
            state = apply_event(state, "rent_payment", rent)
            events.append(_make_event(customer_id, next_timestamp(), "rent_payment", rent, "debit", "housing", before, state))

        if month_day == profile["loan_due_day"]:
            due_amount = profile["income"] * rng.uniform(0.1, 0.15)
            before = replace(state)
            state = apply_event(state, "loan_emi_due", due_amount)
            events.append(_make_event(customer_id, next_timestamp(), "loan_emi_due", due_amount, "debit", "loan", before, state))

            success_probability = _payment_success_probability(state, profile, due_amount, monthly_shock)
            payment_draw = rng.random()
            if payment_draw < success_probability and state.balance > due_amount * 0.65:
                before = replace(state)
                paid_amount = due_amount if payment_draw < success_probability * 0.75 else due_amount * rng.uniform(0.55, 0.9)
                state = apply_event(state, "loan_emi_paid", paid_amount)
                events.append(_make_event(customer_id, next_timestamp(), "loan_emi_paid", paid_amount, "debit", "loan", before, state))
            else:
                before = replace(state)
                state = apply_event(state, "loan_emi_missed", due_amount)
                events.append(_make_event(customer_id, next_timestamp(), "loan_emi_missed", due_amount, "debit", "loan", before, state))
                if rng.random() < profile["recovery_tendency"] * 0.5 and state.balance > due_amount * 0.3:
                    catch_up_amount = due_amount * rng.uniform(0.3, 0.7)
                    before = replace(state)
                    state = apply_event(state, "loan_emi_paid", catch_up_amount)
                    events.append(_make_event(customer_id, next_timestamp(), "loan_emi_paid", catch_up_amount, "debit", "loan", before, state))

        if month_day == profile["credit_due_day"]:
            cc_due = profile["income"] * rng.uniform(0.07, 0.11)
            before = replace(state)
            state = apply_event(state, "credit_card_payment_due", cc_due)
            events.append(_make_event(customer_id, next_timestamp(), "credit_card_payment_due", cc_due, "debit", "credit", before, state))

            success_probability = _payment_success_probability(state, profile, cc_due, monthly_shock)
            payment_draw = rng.random()
            if payment_draw < success_probability and state.balance > cc_due * 0.5:
                before = replace(state)
                paid_amount = cc_due if payment_draw < success_probability * 0.7 else cc_due * rng.uniform(0.45, 0.85)
                state = apply_event(state, "credit_card_payment_made", paid_amount)
                events.append(_make_event(customer_id, next_timestamp(), "credit_card_payment_made", paid_amount, "debit", "credit", before, state))
            else:
                before = replace(state)
                state = apply_event(state, "failed_debit", cc_due)
                events.append(_make_event(customer_id, next_timestamp(), "failed_debit", cc_due, "debit", "credit", before, state))
                if rng.random() < profile["recovery_tendency"] * 0.45 and state.balance > cc_due * 0.25:
                    catch_up_amount = cc_due * rng.uniform(0.25, 0.6)
                    before = replace(state)
                    state = apply_event(state, "credit_card_payment_made", catch_up_amount)
                    events.append(_make_event(customer_id, next_timestamp(), "credit_card_payment_made", catch_up_amount, "debit", "credit", before, state))

        stress = _liquidity_stress(state, profile)
        daily_spend_count = int(rng.integers(0, 4 if stress < 0.55 else 3))
        for _ in range(daily_spend_count):
            spend_base = profile["income"] * spending_pressure / 14
            spend_amount = _sample_amount(rng, spend_base, 0.35 + 0.2 * profile["volatility"])
            event_choice = rng.random()
            if event_choice < 0.5:
                event_type = "card_spend"
                category = "credit"
            elif event_choice < 0.8:
                event_type = "grocery_spend"
                category = "daily_needs"
            else:
                event_type = "transfer_out"
                category = "cash_management"
            direction = "debit"
            before = replace(state)
            state = apply_event(state, event_type, spend_amount)
            events.append(_make_event(customer_id, next_timestamp(), event_type, spend_amount, direction, category, before, state))

        if rng.random() < (0.14 + 0.08 * profile["volatility"]):
            transport = _sample_amount(rng, 24, 0.25)
            before = replace(state)
            state = apply_event(state, "transportation_spend", transport)
            events.append(_make_event(customer_id, next_timestamp(), "transportation_spend", transport, "debit", "transport", before, state))

        if rng.random() < 0.1:
            utility = _sample_amount(rng, 65, 0.2)
            before = replace(state)
            state = apply_event(state, "utility_payment", utility)
            events.append(_make_event(customer_id, next_timestamp(), "utility_payment", utility, "debit", "utilities", before, state))

        if rng.random() < max(0.03, 0.08 * profile["recovery_tendency"]):
            transfer_in = _sample_amount(rng, profile["income"] * 0.08, 0.4)
            before = replace(state)
            state = apply_event(state, "transfer_in", transfer_in)
            events.append(_make_event(customer_id, next_timestamp(), "transfer_in", transfer_in, "credit", "support_network", before, state))

        if rng.random() < (0.07 + 0.05 * stress):
            atm_amount = _sample_amount(rng, 45, 0.3)
            before = replace(state)
            state = apply_event(state, "atm_withdrawal", atm_amount)
            events.append(_make_event(customer_id, next_timestamp(), "atm_withdrawal", atm_amount, "debit", "cash", before, state))

        if state.due_amount > 0 and rng.random() < profile["recovery_tendency"] * 0.2 and state.balance > state.due_amount * 0.2:
            catch_up_amount = state.due_amount * rng.uniform(0.2, 0.5)
            before = replace(state)
            state = apply_event(state, "loan_emi_paid", catch_up_amount)
            events.append(_make_event(customer_id, next_timestamp(), "loan_emi_paid", catch_up_amount, "debit", "loan", before, state))

        if _should_emit_support_contact(state, profile, rng):
            before = replace(state)
            state = apply_event(state, "support_contact", 0.0)
            events.append(_make_event(customer_id, next_timestamp(), "support_contact", 0.0, "debit", "servicing", before, state))

        if state.balance < -50 and rng.random() < 0.8:
            before = replace(state)
            state = apply_event(state, "overdraft_event", abs(state.balance))
            events.append(_make_event(customer_id, next_timestamp(), "overdraft_event", abs(state.balance), "debit", "fees", before, state))

        if state.balance < 0 and rng.random() < 0.18:
            fee_amount = _sample_amount(rng, 18, 0.2)
            before = replace(state)
            state = apply_event(state, "bank_fee_penalty", fee_amount)
            events.append(_make_event(customer_id, next_timestamp(), "bank_fee_penalty", fee_amount, "debit", "fees", before, state))

    return sorted(events, key=lambda item: item.event_timestamp), profile


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
        events, profile = _generate_customer_events(customer_id, archetype, start_date, config, rng)
        rows.extend(event.to_dict() for event in events)
        customers.append(
            {
                "customer_id": customer_id,
                "archetype": archetype,
                "income_band": profile["income_band"],
                "employment_type": profile["employment_type"],
                "region": profile["region"],
                "risk_segment": profile["risk_segment"],
            }
        )

    events_df = pd.DataFrame(rows)
    events_df = _apply_forward_distress_labels(events_df)
    customers_df = pd.DataFrame(customers)
    distress_rates = (
        events_df.groupby("customer_id")["distress_label_30d"].max().rename("customer_distress_label").reset_index()
    )
    customers_df = customers_df.merge(distress_rates, on="customer_id", how="left")
    return events_df, customers_df
