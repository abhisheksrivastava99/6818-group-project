# BankSimFM

BankSimFM is a course-project prototype for retail banking financial distress early warning, inspired by the MarS financial market simulation framework. Instead of modeling market order flow, it models time-ordered customer banking events such as salary credits, bill payments, card spend, missed dues, overdraft events, and support interactions.

The prototype includes:

- a synthetic retail banking event generator
- a shared sequence-data pipeline
- a causal transformer live scorer and forecaster, plus an LSTM baseline benchmark
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
- Transformer as the default live 30-day distress scorer when compatible artifacts exist
- LSTM baseline benchmark retained for comparison in holdout metrics and fairness views
- Public APIs:
  - `score_customer(history, horizon_days=30)`
  - `forecast_customer(history, horizon_days=30)`
  - `simulate_intervention(history, intervention_type, horizon_days=30)`
- Streamlit dashboard pages:
  - Overview
  - Customer Explorer
  - What-If Simulator
  - Collections Prioritization
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
- `82,754` event rows

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

For a no-retrain refresh of simulation summaries only, use:

```bash
PYTHONPATH=src python3 refresh_simulation_metrics.py
```

This reuses the current saved checkpoints and rewrites only `artifacts/simulation_metrics.json`.

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

Live demo:

- [BankSimFM Streamlit App](https://6818-group9-banksimfm.streamlit.app/)

The app supports:

- cohort-level overview and distress distribution
- per-customer event timeline, balance chart, utilization chart, and risk review
- intervention what-if comparison across `30`, `60`, and `90` day horizons
- collections/outreach prioritization with ranked customers, recommended interventions, and projected risk reduction
- top forecasted negative-event summaries and scenario metrics
- portfolio stress monitoring by archetype, income band, employment type, region, and risk segment
- model and governance documentation with fairness summaries
- intervention scenarios that preserve adjusted account state through decoding instead of only swapping an intervention token
- a transformer-first live scoring path that does not require rerunning `train.py` when only the default scorer choice changes


## Current Limitations

- Forecasting is transformer-decoded when compatible trained artifacts exist, but inference still retains heuristic fallbacks if artifacts are missing or stale
- Intervention conditioning is learned directionally inside the synthetic environment, not a causal estimate of real-world treatment effect
- Transformer checkpoint compatibility changes when internal sequence features change, so a fresh `train.py` run is required after major model/pipeline updates
- The prototype is optimized for coursework and demos, not production deployment
- Full training speed depends on local PyTorch device support
- Submission packaging, slide design, and final presentation polish are handled outside the core codebase

