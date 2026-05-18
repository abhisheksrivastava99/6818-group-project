"""Schema definitions for synthetic banking events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional


EVENT_TYPES = [
    "salary_credit",
    "transfer_in",
    "transfer_out",
    "rent_payment",
    "utility_payment",
    "grocery_spend",
    "transportation_spend",
    "loan_emi_due",
    "loan_emi_paid",
    "loan_emi_missed",
    "credit_card_payment_due",
    "credit_card_payment_made",
    "card_spend",
    "atm_withdrawal",
    "bank_fee_penalty",
    "support_contact",
    "failed_debit",
    "overdraft_event",
]


@dataclass
class CustomerEvent:
    customer_id: str
    event_timestamp: datetime
    event_type: str
    amount: float
    amount_direction: str
    category: str
    balance_before: float
    balance_after: float
    credit_limit: float
    credit_utilization: float
    days_to_next_due: Optional[int]
    intervention_flag: str
    distress_label_30d: int

    def to_dict(self) -> Dict[str, object]:
        item = asdict(self)
        item["event_timestamp"] = self.event_timestamp.isoformat()
        return item
