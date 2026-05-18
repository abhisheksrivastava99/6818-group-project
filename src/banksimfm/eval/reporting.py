"""Artifact reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from banksimfm.config import default_config


def load_metrics_summary(root_dir: Path | None = None) -> Dict[str, object]:
    config = default_config(root_dir)
    metrics_path = config.artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())
