from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from banksimfm.config import AppConfig, DataConfig, ModelConfig, ProjectConfig
from banksimfm.data.pipeline import build_datasets
from banksimfm.inference import forecast_customer, score_customer, simulate_intervention
from banksimfm.models.training import train_models


def tiny_config(root_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        root_dir=root_dir,
        data=DataConfig(num_customers=20, history_days=45, max_events_per_customer=128, seed=11),
        model=ModelConfig(
            context_length=64,
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


class BankSimFMTests(unittest.TestCase):
    def test_generated_sequences_are_chronological_and_balances_match(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))
            events = bundle.events.sort_values(["customer_id", "event_timestamp"]).reset_index(drop=True)
            for customer_id, frame in events.groupby("customer_id"):
                self.assertTrue(frame["event_timestamp"].is_monotonic_increasing, customer_id)
                self.assertTrue((frame["credit_utilization"] >= 0).all(), customer_id)
                self.assertTrue((frame["credit_utilization"] <= 1.5).all(), customer_id)

                continuity = frame["balance_after"].shift(1).dropna().reset_index(drop=True)
                current_before = frame["balance_before"].iloc[1:].reset_index(drop=True)
                pd.testing.assert_series_equal(continuity.round(2), current_before.round(2), check_names=False)

    def test_customer_splits_have_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))
            train_ids = set(bundle.splits["train"]["customer_id"].unique())
            val_ids = set(bundle.splits["val"]["customer_id"].unique())
            test_ids = set(bundle.splits["test"]["customer_id"].unique())
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertTrue(train_ids.isdisjoint(test_ids))
            self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_training_saves_metrics_and_models(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            config = tiny_config(Path(tmp_dir))
            artifacts = train_models(config)
            self.assertIn("transformer", artifacts.metrics)
            self.assertIn("lstm", artifacts.metrics)
            self.assertTrue((config.artifacts_dir / "transformer.pt").exists())
            self.assertTrue((config.artifacts_dir / "lstm.pt").exists())
            self.assertTrue((config.artifacts_dir / "metrics.json").exists())

    def test_public_inference_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))
            customer_id = bundle.customers["customer_id"].iloc[0]
            history = bundle.events[bundle.events["customer_id"] == customer_id].sort_values("event_timestamp")

            score = score_customer(history, horizon_days=30)
            self.assertIsInstance(score.distress_probability, float)
            self.assertIsInstance(score.top_drivers, list)
            self.assertIsInstance(score.recent_risk_signals, list)

            forecast = forecast_customer(history, horizon_days=30)
            self.assertGreater(len(forecast.projected_events), 0)
            self.assertEqual(len(forecast.balance_path), len(forecast.utilization_path))

            result = simulate_intervention(history, "temporary_overdraft_buffer", horizon_days=30)
            self.assertIsInstance(result.risk_delta, float)
            self.assertGreater(len(result.scenario_differences), 0)


if __name__ == "__main__":
    unittest.main()
