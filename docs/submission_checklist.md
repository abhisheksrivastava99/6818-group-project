# BankSimFM Submission Checklist

This checklist maps the current workspace against the project requirements in [requirements.md](/Users/abhishek/Desktop/Projects/regtech/requirements.md) and the MH6818 brief in [MH6818_Group_Project.md](/Users/abhishek/Desktop/Projects/regtech/MH6818_Group_Project.md).

## 1. Domain And Innovation Thesis

- [x] Select retail banking as the focus domain.
- [x] Explain why financial foundation models can drive innovation in retail banking.
- [x] Connect the proposal to the MarS paper and explain the adaptation from market simulation to customer-event simulation.
- [ ] Make sure the final report clearly states the innovation thesis in concise business language.
- [ ] Make sure slides include a simple MarS-to-BankSimFM mapping visual.

## 2. Data Strategy And Preparation

### Already Implemented In Code

- [x] Synthetic customer-event dataset generator exists in [src/banksimfm/data/generator.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/generator.py).
- [x] Event schema is implemented in [src/banksimfm/data/schema.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/schema.py).
- [x] Customer-level train/validation/test split is implemented in [src/banksimfm/data/pipeline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/pipeline.py).
- [x] Feature bucketing and history window construction are implemented in [src/banksimfm/data/pipeline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/pipeline.py).
- [x] Packaged demo artifacts exist in [artifacts](/Users/abhishek/Desktop/Projects/regtech/artifacts).

### Still Needed For Submission

- [ ] Final report should describe the real production data a bank would require.
- [ ] Final report should explain how production data would be collected, anonymized, cleaned, and structured.
- [ ] Final report should include preprocessing steps in plain language with an example training sample.
- [ ] State clearly that the prototype uses synthetic data only and contains no PII.
- [x] Fairness-ready customer attributes exist in the synthetic customer table:
  income band, employment type, region, and risk segment.

## 3. Model Architecture And Training Approach

### Already Implemented In Code

- [x] Transformer model exists in [src/banksimfm/models/transformer.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/transformer.py).
- [x] LSTM baseline exists in [src/banksimfm/models/baseline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/baseline.py).
- [x] Training pipeline exists in [src/banksimfm/models/training.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/training.py).
- [x] Early stopping and checkpoint saving are implemented in [src/banksimfm/models/training.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/training.py).
- [x] Holdout metrics are saved to [artifacts/metrics.json](/Users/abhishek/Desktop/Projects/regtech/artifacts/metrics.json).
- [x] Training entrypoint exists in [train.py](/Users/abhishek/Desktop/Projects/regtech/train.py).

### Still Needed In Report Or Slides

- [ ] Explain the transformer architecture in a presentation-friendly diagram.
- [ ] Explain the LSTM baseline and why it is the benchmark.
- [ ] Explain the training objective, batch construction, and evaluation approach in plain language.
- [ ] State infrastructure assumptions such as CPU acceptable, GPU preferred, and Apple Silicon support.

### Still Needed In Code For Stronger Alignment

- [x] Add true autoregressive generation instead of heuristic future-event generation.
- [x] Add intervention conditioning to the model rather than only rule-based intervention policies.
- [x] Add additional heads or objectives for:
  next amount bucket prediction and next balance-delta bucket prediction.
- [ ] If time permits, evaluate multiple random seeds and compare stability.

## 4. Forecasting, Scoring, And Simulation Interfaces

### Already Implemented In Code

- [x] `score_customer(history, horizon_days=30)` exists in [src/banksimfm/inference.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/inference.py).
- [x] `forecast_customer(history, horizon_days=30)` exists in [src/banksimfm/inference.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/inference.py).
- [x] `simulate_intervention(history, intervention_type, horizon_days=30)` exists in [src/banksimfm/inference.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/inference.py).
- [x] Deterministic account-state engine exists in [src/banksimfm/sim/engine.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/sim/engine.py).

### Still Needed For Stronger Requirement Coverage

- [x] Make forecasting model-driven instead of rule-driven.
- [x] Make intervention simulation learned or conditioned by the sequence model.
- [x] Add a clearer summary of top forecasted negative events for each scenario.

## 5. Evaluation

### Already Implemented In Code

- [x] AUC, precision, recall, F1, and accuracy are computed and saved.
- [x] Transformer and LSTM are compared on the same holdout target.
- [x] Automated tests run successfully.

### Still Needed In Code Or Analysis

- [x] Add early-warning metrics such as lead time before distress events.
- [x] Add false-positive analysis for stable customers.
- [x] Add simulation quality metrics:
  plausibility of event mix, balance realism, and similarity to generator assumptions.
- [x] Add intervention usefulness metrics:
  risk reduction frequency, material scenario changes, and per-intervention comparison.
- [x] Add repeated-run stability analysis if simulation remains stochastic.
- [x] Add fairness breakdowns across customer segments.

## 6. Downstream Applications

### Already Covered

- [x] Early distress warning is supported.
- [x] Intervention testing is supported.
- [x] Synthetic data generation for sandbox experimentation is supported.

### Still Needed In Report

- [ ] Describe collections prioritization use case in detail.
- [ ] Describe portfolio stress monitoring use case in detail.
- [ ] Describe how these applications translate into operational or economic decisions.

### Still Needed In Code If You Want Stronger Demo Coverage

- [ ] Add a collections prioritization or ranked-customer view in the dashboard.
- [x] Add portfolio-level stress or cohort scenario monitoring in the dashboard.

## 7. Governance, Risk, And Controls

### Already Partly Covered

- [x] Synthetic-only data design reduces privacy risk.
- [x] Heuristic top drivers and timeline review provide some explainability.
- [x] The prototype clearly acts as a demo rather than autonomous decisioning.

### Still Needed In Report

- [ ] Write a dedicated governance section covering privacy, fairness, explainability, reliability, and operational risk.
- [ ] State that the prototype should not be used for live credit decisions.
- [ ] Explain why intervention results are directional, not causal proof.
- [ ] Propose human review and monitoring controls for any real deployment.

### Still Needed In Code Or UI

- [x] Add a dashboard section or page for governance notes.
- [x] Add explicit fairness note, privacy note, explainability note, and reliability note to the UI.
- [x] Add empirical fairness evaluation if customer segment fields are available.

## 8. Economic Value Measurement

### Still Needed In Report

- [ ] Define value drivers:
  avoided missed-payment loss, reduced collections cost, retention uplift, operational savings.
- [ ] Define a scenario-based methodology to estimate value.
- [ ] Make clear that value is proposed, not realized.
- [ ] Link each downstream application to a business decision and value mechanism.

## 9. Streamlit Frontend

### Already Implemented In Code

- [x] Streamlit entrypoint exists in [app.py](/Users/abhishek/Desktop/Projects/regtech/app.py).
- [x] Overview page exists.
- [x] Customer Explorer page exists.
- [x] What-If Simulator page exists.
- [x] KPI cards, tables, and charts are already present.

### Still Needed For Full Requirement Match

- [x] Add a Model and Governance page or section.
- [x] Add an explicit synthetic demo data banner or label.
- [x] Add a clearer balance chart in Customer Explorer.
- [x] Add a focused negative-event summary in What-If Simulator.
- [x] Add architecture summary and transformer-vs-LSTM rationale to the UI.

## 10. Testing And Validation

- [x] Unit tests exist in [tests/test_banksimfm.py](/Users/abhishek/Desktop/Projects/regtech/tests/test_banksimfm.py).
- [x] Data sanity tests exist in [tests/test_data_distribution.py](/Users/abhishek/Desktop/Projects/regtech/tests/test_data_distribution.py).
- [x] Current test suite passes locally.
- [ ] If you add new dashboard or evaluation features, add tests for them where practical.

## 11. Final Deliverables

### Code And Data

- [x] Sample synthetic data exists.
- [x] Prototype code exists.
- [x] Training artifacts exist.

### Still Needed Outside Core Code

- [ ] Final report document.
- [ ] PowerPoint slide deck.
- [ ] Appendix if needed.
- [ ] Zip packaging of all deliverables for submission.

## 12. Recommended Submission Priority

### Must Finish

- [ ] Final report write-up for data strategy, model rationale, governance, economic value, assumptions, and downstream applications.
- [ ] Slides that clearly explain the MarS alignment, architecture, and outcomes.
- [x] Dashboard/UI updates for Model and Governance content for full frontend requirement coverage.

### High Value If Time Permits

- [x] Add early-warning and intervention evaluation metrics.
- [ ] Add collections prioritization and ranked-customer actioning view.
- [x] Add portfolio stress views.
- [x] Add customer segment fields to support fairness evaluation.

### Stretch Goal

- [x] Replace heuristic forecasting with transformer-driven autoregressive simulation.
