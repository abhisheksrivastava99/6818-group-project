"""Central configuration for the prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    num_customers: int = 300
    history_days: int = 120
    max_events_per_customer: int = 256
    seed: int = 7
    archetypes: List[str] = field(
        default_factory=lambda: [
            "stable_salaried",
            "volatile_income",
            "high_obligation",
            "rising_utilization",
            "near_distress",
        ]
    )


@dataclass
class ModelConfig:
    context_length: int = 256
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 8
    patience: int = 4
    forecast_steps: int = 12


@dataclass
class AppConfig:
    default_horizon_days: int = 30
    allowed_horizons: List[int] = field(default_factory=lambda: [30, 60, 90])
    distressed_threshold: float = 0.5


@dataclass
class ProjectConfig:
    root_dir: Path
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def config_summary(self) -> Dict[str, object]:
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "app": self.app.__dict__,
        }


def default_config(root_dir: Path | None = None) -> ProjectConfig:
    base_dir = root_dir or Path(__file__).resolve().parents[2]
    return ProjectConfig(root_dir=base_dir)
