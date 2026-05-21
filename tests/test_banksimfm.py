from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from banksimfm.config import AppConfig, DataConfig, ModelConfig, ProjectConfig
from banksimfm.data.pipeline import build_datasets
from banksimfm.app.dashboard import (
    _build_collections_prioritization_frame,
    _filter_collections_prioritization_frame,
    _priority_tier,
)
from banksimfm.inference import forecast_customer, score_customer, simulate_intervention
from banksimfm.models.training import refresh_simulation_metrics, train_models
from banksimfm.models.transformer import CausalEventTransformer
from banksimfm.data.pipeline import AMOUNT_BINS, BALANCE_BINS, DELTA_BINS, DUE_BINS, GAP_BINS, UTIL_BINS
from banksimfm.sim.engine import AccountState, apply_intervention_policy
from banksimfm.sim.scenario import apply_state_to_history, decode_forecast, history_to_state


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


class FixedTransformer(torch.nn.Module):
    def __init__(self, vocab_size: int, amount_size: int, delta_size: int, event_id: int, amount_id: int, delta_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.amount_size = amount_size
        self.delta_size = delta_size
        self.event_id = event_id
        self.amount_id = amount_id
        self.delta_id = delta_id

    def forward(self, seq_tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = seq_tokens.shape[0]
        event_logits = torch.full((batch_size, self.vocab_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        amount_logits = torch.full((batch_size, self.amount_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        delta_logits = torch.full((batch_size, self.delta_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        distress_logits = torch.zeros((batch_size,), dtype=torch.float32, device=seq_tokens.device)
        event_logits[:, self.event_id] = 0.0
        amount_logits[:, self.amount_id] = 0.0
        delta_logits[:, self.delta_id] = 0.0
        return {
            "next_event_logits": event_logits,
            "next_amount_logits": amount_logits,
            "next_balance_delta_logits": delta_logits,
            "distress_logits": distress_logits,
        }


class InterventionAwareTransformer(torch.nn.Module):
    def __init__(self, vocab_size: int, amount_size: int, delta_size: int, baseline_event_id: int, intervention_event_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.amount_size = amount_size
        self.delta_size = delta_size
        self.baseline_event_id = baseline_event_id
        self.intervention_event_id = intervention_event_id

    def forward(self, seq_tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = seq_tokens.shape[0]
        intervention_active = seq_tokens[:, :, 6].max(dim=1).values > 0
        event_logits = torch.full((batch_size, self.vocab_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        amount_logits = torch.full((batch_size, self.amount_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        delta_logits = torch.full((batch_size, self.delta_size), -1e9, dtype=torch.float32, device=seq_tokens.device)
        distress_logits = torch.where(
            intervention_active,
            torch.full((batch_size,), -2.0, dtype=torch.float32, device=seq_tokens.device),
            torch.full((batch_size,), 1.5, dtype=torch.float32, device=seq_tokens.device),
        )
        event_logits[:, self.baseline_event_id] = 0.0
        event_logits[intervention_active, :] = -1e9
        event_logits[intervention_active, self.intervention_event_id] = 0.0
        amount_logits[:, min(6, self.amount_size - 1)] = 0.0
        delta_logits[:, min(6, self.delta_size - 1)] = 0.0
        return {
            "next_event_logits": event_logits,
            "next_amount_logits": amount_logits,
            "next_balance_delta_logits": delta_logits,
            "distress_logits": distress_logits,
        }


class ConstantLSTM(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, dense_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.full((dense_features.shape[0],), self.logit, dtype=torch.float32, device=dense_features.device)


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
            self.assertTrue((config.artifacts_dir / "simulation_metrics.json").exists())
            self.assertTrue((config.artifacts_dir / "fairness_metrics.json").exists())

    def test_customer_metadata_and_transformer_heads(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, datasets = build_datasets(tiny_config(Path(tmp_dir)))
            for column in ["income_band", "employment_type", "region", "risk_segment"]:
                self.assertIn(column, bundle.customers.columns)

            sample = datasets["train"][0]
            self.assertEqual(sample["seq_tokens"].shape[-1], 7)
            self.assertEqual(sample["dense_features"].shape[-1], 8)
            transformer = CausalEventTransformer(
                vocab_size=len(bundle.encoders.event_type_to_id) + 1,
                bucket_sizes={
                    "amount": len(AMOUNT_BINS),
                    "balance": len(BALANCE_BINS),
                    "util": len(UTIL_BINS),
                    "due": len(DUE_BINS),
                    "gap": len(GAP_BINS),
                    "delta": len(DELTA_BINS),
                    "intervention": len(bundle.encoders.intervention_type_to_id),
                },
                hidden_size=64,
                num_layers=1,
                num_heads=4,
                dropout=0.1,
            )
            outputs = transformer(sample["seq_tokens"].unsqueeze(0), sample["mask"].unsqueeze(0))
            self.assertIn("next_event_logits", outputs)
            self.assertIn("next_amount_logits", outputs)
            self.assertIn("next_balance_delta_logits", outputs)
            self.assertIn("distress_logits", outputs)

    def test_intervention_augmented_samples_and_stateful_decoding(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            config = tiny_config(Path(tmp_dir))
            config.model.intervention_augmentation_rate = 1.0
            config.model.intervention_augmentation_steps = 2
            bundle, datasets = build_datasets(config)

            augmented_samples = [sample for sample in datasets["train"].samples if float(sample["lstm_sample_weight"].item()) == 0.0]
            self.assertGreater(len(augmented_samples), 0)
            self.assertTrue(any(int(sample["seq_tokens"][:, -1].max().item()) > 0 for sample in augmented_samples))

            history = pd.DataFrame(
                [
                    {
                        "customer_id": "C_TEST",
                        "event_timestamp": pd.Timestamp("2025-01-01"),
                        "event_type": "loan_emi_due",
                        "amount": 400.0,
                        "amount_direction": "debit",
                        "category": "loan",
                        "balance_before": 1600.0,
                        "balance_after": 1600.0,
                        "credit_limit": 2000.0,
                        "credit_utilization": 0.2,
                        "due_amount_feature": 400.0,
                        "days_to_next_due": 1,
                        "intervention_flag": "none",
                    },
                    {
                        "customer_id": "C_TEST",
                        "event_timestamp": pd.Timestamp("2025-01-02"),
                        "event_type": "card_spend",
                        "amount": 120.0,
                        "amount_direction": "debit",
                        "category": "credit",
                        "balance_before": 1600.0,
                        "balance_after": 1600.0,
                        "credit_limit": 2000.0,
                        "credit_utilization": 0.26,
                        "due_amount_feature": 400.0,
                        "days_to_next_due": 1,
                        "intervention_flag": "none",
                    },
                ]
            )
            base_state = history_to_state(history)
            adjusted_state = apply_intervention_policy(base_state, "installment_restructure")
            adjusted_history = apply_state_to_history(history, adjusted_state, "installment_restructure")

            dummy = FixedTransformer(
                vocab_size=len(bundle.encoders.event_type_to_id) + 1,
                amount_size=len(AMOUNT_BINS),
                delta_size=len(DELTA_BINS),
                event_id=bundle.encoders.event_type_to_id["loan_emi_paid"],
                amount_id=6,
                delta_id=6,
            )
            baseline = decode_forecast(dummy, history, "none", 30, torch.device("cpu"), context_length=64, strategy="greedy", starting_state=base_state)
            intervention = decode_forecast(
                dummy,
                adjusted_history,
                "installment_restructure",
                30,
                torch.device("cpu"),
                context_length=64,
                strategy="greedy",
                starting_state=adjusted_state,
            )
            self.assertNotEqual(baseline["projected_events"][0]["amount"], intervention["projected_events"][0]["amount"])
            self.assertGreater(intervention["balance_path"][-1], baseline["balance_path"][-1])

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

    def test_simulate_intervention_uses_transformer_scenario_rescoring(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))
            customer_id = bundle.customers["customer_id"].iloc[0]
            history = bundle.events[bundle.events["customer_id"] == customer_id].sort_values("event_timestamp").tail(20)
            transformer = InterventionAwareTransformer(
                vocab_size=len(bundle.encoders.event_type_to_id) + 1,
                amount_size=len(AMOUNT_BINS),
                delta_size=len(DELTA_BINS),
                baseline_event_id=bundle.encoders.event_type_to_id["card_spend"],
                intervention_event_id=bundle.encoders.event_type_to_id["credit_card_payment_made"],
            )
            fake_loaded = {
                "device": torch.device("cpu"),
                "transformer": transformer,
                "lstm": ConstantLSTM(logit=-10.0),
                "thresholds": {"transformer": 0.5, "lstm": 0.5},
            }
            with patch("banksimfm.inference._load_trained_models", return_value=fake_loaded):
                result = simulate_intervention(history, "temporary_overdraft_buffer", horizon_days=30)
            self.assertNotEqual(result.baseline.risk.distress_probability, result.intervention.risk.distress_probability)
            self.assertNotEqual(result.risk_delta, 0.0)

    def test_score_customer_prefers_transformer_when_available(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))
            customer_id = bundle.customers["customer_id"].iloc[0]
            history = bundle.events[bundle.events["customer_id"] == customer_id].sort_values("event_timestamp").tail(20)
            fake_loaded = {
                "device": torch.device("cpu"),
                "transformer": InterventionAwareTransformer(
                    vocab_size=len(bundle.encoders.event_type_to_id) + 1,
                    amount_size=len(AMOUNT_BINS),
                    delta_size=len(DELTA_BINS),
                    baseline_event_id=bundle.encoders.event_type_to_id["card_spend"],
                    intervention_event_id=bundle.encoders.event_type_to_id["credit_card_payment_made"],
                ),
                "lstm": ConstantLSTM(logit=-10.0),
                "thresholds": {"transformer": 0.8, "lstm": 0.5},
            }
            with patch("banksimfm.inference._load_trained_models", return_value=fake_loaded):
                score = score_customer(history, horizon_days=30)
            self.assertGreater(score.distress_probability, 0.8)
            self.assertTrue(score.distress_label)

    def test_refresh_simulation_metrics_uses_saved_artifacts_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            config = tiny_config(Path(tmp_dir))
            train_models(config)
            fairness_path = config.artifacts_dir / "fairness_metrics.json"
            fairness_before = fairness_path.read_text()
            metrics_path = config.artifacts_dir / "simulation_metrics.json"
            before_mtime = metrics_path.stat().st_mtime

            with patch("banksimfm.models.training.train_models", side_effect=AssertionError("refresh should not retrain")):
                refreshed = refresh_simulation_metrics(config)

            fairness_after = fairness_path.read_text()
            self.assertEqual(fairness_before, fairness_after)
            self.assertGreaterEqual(metrics_path.stat().st_mtime, before_mtime)
            self.assertIn("intervention_usefulness", refreshed)
            repo_root = Path(__file__).resolve().parents[1]
            self.assertTrue((repo_root / "refresh_simulation_metrics.py").exists())
            written = json.loads(metrics_path.read_text())
            self.assertEqual(written["stability"]["repeat_runs"], 10)

    def test_priority_tier_rules(self) -> None:
        self.assertEqual(_priority_tier(0.60, 0.10), "High")
        self.assertEqual(_priority_tier(0.35, 0.01), "Medium")
        self.assertEqual(_priority_tier(0.20, 0.10), "Monitor")
        self.assertEqual(_priority_tier(0.55, -0.01), "Monitor")

    def test_collections_prioritization_frame_contains_ranked_recommendations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            bundle, _ = build_datasets(tiny_config(Path(tmp_dir)))

            def fake_score(history: pd.DataFrame, horizon_days: int = 30):
                customer_id = history.iloc[-1]["customer_id"]
                score_map = {
                    "CUST_0000": 0.72,
                    "CUST_0001": 0.48,
                }
                probability = score_map.get(customer_id, 0.15)
                return type(
                    "ScoreResultStub",
                    (),
                    {
                        "distress_probability": probability,
                        "distress_label": probability >= 0.5,
                        "top_drivers": ["recent low balances"],
                        "recent_risk_signals": ["synthetic signal"],
                    },
                )()

            def fake_simulate(history: pd.DataFrame, intervention_type: str, horizon_days: int = 30):
                customer_id = history.iloc[-1]["customer_id"]
                base_map = {"CUST_0000": 0.72, "CUST_0001": 0.48}
                intervention_map = {
                    ("CUST_0000", "reminder"): 0.70,
                    ("CUST_0000", "due_date_shift_7d"): 0.61,
                    ("CUST_0000", "temporary_overdraft_buffer"): 0.40,
                    ("CUST_0000", "installment_restructure"): 0.31,
                    ("CUST_0001", "reminder"): 0.47,
                    ("CUST_0001", "due_date_shift_7d"): 0.45,
                    ("CUST_0001", "temporary_overdraft_buffer"): 0.52,
                    ("CUST_0001", "installment_restructure"): 0.43,
                }
                baseline_probability = base_map.get(customer_id, 0.15)
                intervention_probability = intervention_map.get((customer_id, intervention_type), baseline_probability)
                baseline_risk = type("ScoreResultStub", (), {"distress_probability": baseline_probability})()
                intervention_risk = type("ScoreResultStub", (), {"distress_probability": intervention_probability})()
                return type(
                    "SimulationResultStub",
                    (),
                    {
                        "baseline": type("SimulationSideStub", (), {"risk": baseline_risk})(),
                        "intervention": type("SimulationSideStub", (), {"risk": intervention_risk})(),
                    },
                )()

            subset_customer_ids = bundle.customers["customer_id"].head(3).tolist()
            subset_customers = bundle.customers[bundle.customers["customer_id"].isin(subset_customer_ids)].copy()
            subset_events = bundle.events[bundle.events["customer_id"].isin(subset_customer_ids)].copy()
            with patch("banksimfm.app.dashboard.score_customer", side_effect=fake_score), patch(
                "banksimfm.app.dashboard.simulate_intervention", side_effect=fake_simulate
            ):
                frame = _build_collections_prioritization_frame(subset_customers, subset_events, horizon_days=30)

            self.assertFalse(frame.empty)
            self.assertEqual(frame.iloc[0]["customer_id"], "CUST_0000")
            self.assertEqual(frame.iloc[0]["recommended_intervention"], "installment_restructure")
            self.assertGreater(frame.iloc[0]["predicted_risk_reduction"], 0.0)
            self.assertIn("current_risk", frame.columns)
            self.assertIn("projected_post_intervention_risk", frame.columns)
            self.assertIn("priority_tier", frame.columns)

    def test_collections_prioritization_filters_reduce_ranked_set(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "customer_id": "A",
                    "archetype": "stable_salaried",
                    "income_band": "middle",
                    "employment_type": "full_time",
                    "region": "north",
                    "risk_segment": "stable",
                    "current_risk": 0.20,
                },
                {
                    "customer_id": "B",
                    "archetype": "high_obligation",
                    "income_band": "low",
                    "employment_type": "gig",
                    "region": "south",
                    "risk_segment": "emerging_risk",
                    "current_risk": 0.60,
                },
            ]
        )
        filtered = _filter_collections_prioritization_frame(
            frame,
            archetype="high_obligation",
            income_band="All",
            employment_type="All",
            region="All",
            risk_segment="All",
            min_current_risk=0.30,
        )
        self.assertEqual(filtered["customer_id"].tolist(), ["B"])


if __name__ == "__main__":
    unittest.main()
