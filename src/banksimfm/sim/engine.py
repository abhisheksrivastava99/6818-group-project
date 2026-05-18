"""Deterministic account-state engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict


@dataclass
class AccountState:
    balance: float
    credit_limit: float
    utilized_credit: float
    missed_payments: int = 0
    overdraft_events: int = 0
    failed_debits: int = 0
    low_balance_streak: int = 0
    due_amount: float = 0.0
    due_in_days: int = 15
    last_income_amount: float = 0.0
    intervention_flag: str = "none"

    @property
    def credit_utilization(self) -> float:
        if self.credit_limit <= 0:
            return 0.0
        return max(0.0, min(1.5, self.utilized_credit / self.credit_limit))


def apply_event(state: AccountState, event_type: str, amount: float) -> AccountState:
    """Apply one event to account state."""
    next_state = replace(state)
    amount = float(amount)

    if event_type in {"salary_credit", "transfer_in"}:
        next_state.balance += amount
        next_state.last_income_amount = amount
    elif event_type in {"transfer_out", "rent_payment", "utility_payment", "grocery_spend", "transportation_spend", "atm_withdrawal", "bank_fee_penalty"}:
        next_state.balance -= amount
    elif event_type == "card_spend":
        next_state.utilized_credit += amount
    elif event_type == "credit_card_payment_made":
        pay = min(amount, next_state.utilized_credit)
        next_state.balance -= pay
        next_state.utilized_credit -= pay
    elif event_type == "credit_card_payment_due":
        next_state.due_amount += amount
        next_state.due_in_days = 5
    elif event_type == "loan_emi_due":
        next_state.due_amount += amount
        next_state.due_in_days = 3
    elif event_type == "loan_emi_paid":
        next_state.balance -= amount
        next_state.due_amount = max(0.0, next_state.due_amount - amount)
    elif event_type == "loan_emi_missed":
        next_state.missed_payments += 1
        next_state.due_amount += amount
    elif event_type == "failed_debit":
        next_state.failed_debits += 1
        next_state.balance -= min(10.0, amount * 0.02)
    elif event_type == "overdraft_event":
        next_state.overdraft_events += 1
    elif event_type == "support_contact":
        pass

    if next_state.balance < 200:
        next_state.low_balance_streak += 1
    else:
        next_state.low_balance_streak = 0

    if next_state.balance < 0:
        next_state.overdraft_events += 1

    next_state.due_in_days = max(0, next_state.due_in_days - 1)
    return next_state


def apply_intervention_policy(state: AccountState, intervention_type: str) -> AccountState:
    """Apply simple intervention rules for what-if simulation."""
    updated = replace(state, intervention_flag=intervention_type)

    if intervention_type == "reminder":
        updated.due_in_days = max(1, updated.due_in_days - 1)
    elif intervention_type == "due_date_shift_7d":
        updated.due_in_days += 7
    elif intervention_type == "temporary_overdraft_buffer":
        updated.balance += 250.0
    elif intervention_type == "installment_restructure":
        updated.due_amount *= 0.7
        updated.due_in_days += 10

    return updated


def state_risk_factors(state: AccountState) -> Dict[str, float]:
    """Return normalized rule-based risk factors for heuristics."""
    return {
        "low_balance": min(1.0, max(0.0, (500.0 - state.balance) / 500.0)),
        "high_utilization": min(1.0, state.credit_utilization),
        "missed_payments": min(1.0, state.missed_payments / 3.0),
        "failed_debits": min(1.0, state.failed_debits / 3.0),
        "overdrafts": min(1.0, state.overdraft_events / 3.0),
        "income_instability": 0.4 if state.last_income_amount < 1500 else 0.1,
    }
