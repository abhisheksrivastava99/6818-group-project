from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from banksimfm.config import AppConfig, DataConfig, ModelConfig, ProjectConfig
from banksimfm.data.pipeline import build_datasets


def medium_config(root_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        root_dir=root_dir,
        data=DataConfig(num_customers=200, history_days=90, max_events_per_customer=192, seed=7),
        model=ModelConfig(
            context_length=128,
            hidden_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.1,
            batch_size=16,
            learning_rate=1e-3,
            max_epochs=1,
            patience=1,
            forecast_steps=8,
        ),
        app=AppConfig(),
    )


class DataDistributionTests(unittest.TestCase):
    def test_archetype_distress_rates_are_ordered_but_not_extreme(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(medium_config(Path(tmp_dir)))
            rates = bundle.customers.groupby("archetype")["customer_distress_label"].mean()

            self.assertLess(rates["stable_salaried"], rates["near_distress"])
            self.assertLess(rates["stable_salaried"], rates["high_obligation"])
            self.assertLess(rates["stable_salaried"], rates["rising_utilization"])
            self.assertLess(rates["stable_salaried"], rates["volatile_income"])

            for archetype, rate in rates.items():
                self.assertGreaterEqual(rate, 0.05, archetype)
                self.assertLess(rate, 0.95, archetype)

    def test_split_positive_rates_and_sequence_lengths_have_variation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(medium_config(Path(tmp_dir)))
            positive_rates = {name: float(frame["distress_label_30d"].mean()) for name, frame in bundle.splits.items()}
            event_counts = bundle.events.groupby("customer_id").size()
            balance_std = float(bundle.events.groupby("customer_id")["balance_after"].mean().std())

            for split_name, rate in positive_rates.items():
                self.assertGreater(rate, 0.05, split_name)
                self.assertLess(rate, 0.9, split_name)

            self.assertGreater(int(event_counts.nunique()), 10)
            self.assertGreater(balance_std, 100.0)


if __name__ == "__main__":
    unittest.main()
