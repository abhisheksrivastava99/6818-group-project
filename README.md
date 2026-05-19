# BankSimFM

BankSimFM is a course-project prototype for retail banking financial distress early warning, inspired by the MarS financial market simulation framework. Instead of modeling market order flow, it models time-ordered customer banking events such as salary credits, bill payments, card spend, missed dues, overdraft events, and support interactions.

The prototype includes:

- a synthetic retail banking event generator
- a shared sequence-data pipeline
- a causal transformer forecaster and LSTM baseline scorer
- a deterministic intervention simulator
- a Streamlit analyst dashboard

## Repository Layout

```text
regtech/
├── app.py
├── train.py
├── artifacts/
├── configs/
├── src/banksimfm/
│   ├── app/
│   ├── data/
│   ├── eval/
│   ├── models/
│   ├── sim/
│   ├── inference.py
│   ├── runtime.py
│   └── types.py
├── tests/
├── requirements.md
└── README.md
```

## What’s Implemented

- Synthetic event generation for five customer archetypes:
  - stable salaried
  - volatile income
  - high obligation
  - rising utilization
  - near distress
- Canonical event schema with chronological account-state consistency
- Customer-level train/validation/test splits
- Shared history windowing for both sequence models
- Internal due-amount state tracking to preserve repayment and restructuring context
- Transformer training for:
  - next-event prediction
  - next amount-bucket prediction
  - next balance-delta-bucket prediction
  - distress classification
- Intervention-conditioned transformer forecasting with explicit state handoff
- LSTM baseline for primary 30-day distress prediction
- Public APIs:
  - `score_customer(history, horizon_days=30)`
  - `forecast_customer(history, horizon_days=30)`
  - `simulate_intervention(history, intervention_type, horizon_days=30)`
- Streamlit dashboard pages:
  - Overview
  - Customer Explorer
  - What-If Simulator
  - Model And Governance
- Portfolio stress monitoring and fairness-ready customer metadata
- Evaluation artifacts for:
  - holdout classification metrics
  - simulation quality and intervention usefulness
  - subgroup fairness breakdowns
- Automated tests for data integrity, split leakage, training artifacts, and inference contract

## Requirements

- Python 3.9+
- macOS, Linux, or Windows
- Recommended Python packages are listed in:
  - [requirements.txt](/Users/abhishek/Desktop/Projects/regtech/requirements.txt)
  - [pyproject.toml](/Users/abhishek/Desktop/Projects/regtech/pyproject.toml)

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Generated Data

Synthetic demo data has already been generated and stored in [artifacts](/Users/abhishek/Desktop/Projects/regtech/artifacts):

- `demo_events.csv`
- `demo_customers.csv`
- `demo_metadata.json`

At the current default config, this is roughly:

- `300` synthetic customers
- `82,391` event rows

The tuned generator now uses softer, probabilistic archetypes with more customer-level variation, recovery behavior, and less deterministic distress pathways.

## Training

Run model training from the project root:

```bash
PYTHONPATH=src python3 train.py
```

This will generate:

- `artifacts/transformer.pt`
- `artifacts/lstm.pt`
- `artifacts/metrics.json`
- `artifacts/simulation_metrics.json`
- `artifacts/fairness_metrics.json`

The training pipeline:

- builds synthetic datasets
- constructs customer-history windows
- uses multi-step intervention-augmented continuations for transformer conditioning
- trains the transformer and LSTM with verbose per-epoch logging
- evaluates both models on validation and test splits
- saves model checkpoints and metric summaries
- computes fairness, early-warning, simulation-realism, stability, and intervention-usefulness summaries

Current tuned defaults include:

- transformer learning rate: `5e-4`
- LSTM learning rate: `1e-3`
- max epochs: `12`
- patience: `5`
- intervention augmentation rate: `0.30`
- intervention augmentation steps: `3`
- transformer distress loss weight: `1.5`

## Streamlit App

Launch the analyst dashboard with:

```bash
PYTHONPATH=src streamlit run app.py
```

The app supports:

- cohort-level overview and distress distribution
- per-customer event timeline, balance chart, utilization chart, and risk review
- intervention what-if comparison across `30`, `60`, and `90` day horizons
- top forecasted negative-event summaries and scenario metrics
- portfolio stress monitoring by archetype, income band, employment type, region, and risk segment
- model and governance documentation with fairness summaries
- intervention scenarios that preserve adjusted account state through decoding instead of only swapping an intervention token

## Apple Silicon Note

The code now prefers devices in this order:

1. `mps`
2. `cuda`
3. `cpu`

That logic lives in [src/banksimfm/runtime.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/runtime.py).

To check what your current environment will use:

```bash
PYTHONPATH=src python3 -c "from banksimfm.runtime import resolve_device; print(resolve_device())"
```

If this prints `cpu` on a Mac with Apple Silicon, your current PyTorch install likely does not expose MPS in that interpreter yet. The project will still run, but training will use CPU.

## Tests

Run the automated tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The current tests cover:

- chronological event generation
- balance continuity across events
- customer-split leakage checks
- archetype distress-rate sanity and split-label distribution checks
- training artifact creation
- public inference API behavior

## Main Files

- [src/banksimfm/data/generator.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/generator.py): synthetic event generation
- [src/banksimfm/data/pipeline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/pipeline.py): preprocessing, bucketing, window construction, customer splits
- [src/banksimfm/models/training.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/training.py): training loops and evaluation
- [src/banksimfm/inference.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/inference.py): public scoring, forecasting, and simulation APIs
- [src/banksimfm/app/dashboard.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/app/dashboard.py): Streamlit dashboard
- [requirements.md](/Users/abhishek/Desktop/Projects/regtech/requirements.md): project requirements specification

## Current Limitations

- Forecasting is transformer-decoded when compatible trained artifacts exist, but inference still retains heuristic fallbacks if artifacts are missing or stale
- Intervention conditioning is learned directionally inside the synthetic environment, not a causal estimate of real-world treatment effect
- Transformer checkpoint compatibility changes when internal sequence features change, so a fresh `train.py` run is required after major model/pipeline updates
- The prototype is optimized for coursework and demos, not production deployment
- Full training speed depends on local PyTorch device support
- Final report, slide deck, and business-value write-up still need to be completed outside the codebase

## Project Context

This repository is designed to support the MH6818 FinTech Innovation with AI group project and the BankSimFM requirements in [requirements.md](/Users/abhishek/Desktop/Projects/regtech/requirements.md).
