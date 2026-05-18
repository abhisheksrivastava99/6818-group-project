"""Shared public result types for BankSimFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ScoreResult:
    distress_probability: float
    distress_label: bool
    top_drivers: List[str]
    recent_risk_signals: List[str]


@dataclass
class ForecastResult:
    projected_events: List[Dict[str, Any]]
    balance_path: List[float]
    utilization_path: List[float]
    forecast_summary: Dict[str, Any]


@dataclass
class SimulationSide:
    risk: ScoreResult
    forecast: ForecastResult


@dataclass
class SimulationResult:
    baseline: SimulationSide
    intervention: SimulationSide
    risk_delta: float
    scenario_differences: List[str]
