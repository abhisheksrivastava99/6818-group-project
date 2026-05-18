"""Data utilities for BankSimFM."""

from .generator import generate_synthetic_dataset
from .pipeline import build_datasets, load_or_create_demo_bundle
from .schema import EVENT_TYPES, CustomerEvent

__all__ = [
    "CustomerEvent",
    "EVENT_TYPES",
    "build_datasets",
    "generate_synthetic_dataset",
    "load_or_create_demo_bundle",
]
