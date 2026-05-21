from __future__ import annotations

import json

from banksimfm.config import default_config
from banksimfm.data.pipeline import load_or_create_demo_bundle
from banksimfm.models.training import load_saved_models, refresh_simulation_metrics


def main() -> None:
    config = default_config()
    load_or_create_demo_bundle(config)
    loaded = load_saved_models(config)
    if loaded is None:
        raise SystemExit("Compatible saved artifacts were not found. Run `PYTHONPATH=src python3 train.py` first.")

    simulation_metrics = refresh_simulation_metrics(config)
    output_path = config.artifacts_dir / "simulation_metrics.json"

    print(f"Updated {output_path}")
    print("Early warning:")
    print(json.dumps(simulation_metrics["early_warning"], indent=2))
    print("Intervention usefulness:")
    print(json.dumps(simulation_metrics["intervention_usefulness"]["per_intervention"], indent=2))


if __name__ == "__main__":
    main()
