# BankSimFM Final Report Draft

**Course:** MH6818 FinTech Innovation with AI  
**Project Title:** BankSimFM: A MarS-Inspired Retail Banking Financial Distress Early-Warning Simulator  
**Group Number:** `[Insert Group Number]`  
**Team Members:** `[Insert Member Names]`  
**Submission Date:** `[Insert Submission Date]`

## Executive Summary

Retail banks often identify customer financial distress only after a harmful event has already occurred, such as a missed payment, failed debit, or overdraft. This project proposes **BankSimFM**, a MarS-inspired retail banking sequence-modeling prototype that treats a customer's financial life as a time-ordered event stream rather than a static tabular snapshot. The system is designed to estimate 30-day distress risk, forecast likely future account behavior, and simulate intervention-conditioned outcomes.

BankSimFM adapts the logic of the MarS financial market simulation framework from market-order trajectories to customer financial-event trajectories. Instead of order flow, the model learns sequences of salary credits, bill payments, card spend, repayment events, overdraft activity, and support interactions. The prototype combines a causal transformer, an LSTM benchmark, a deterministic account-state engine, and a Streamlit dashboard for analyst-facing what-if analysis.

The latest trained system shows that the **transformer is now the strongest primary live scorer**, with test AUC `0.8342` and test F1 `0.5828`, compared with LSTM test AUC `0.8126` and test F1 `0.5126`. The project further supports intervention simulation, portfolio stress monitoring, fairness review, and collections prioritization. At the same time, the project remains a **synthetic-data, course-project prototype** rather than a production decision engine.

The main contribution of the project is demonstrating that a MarS-style financial foundation model can be adapted from markets to retail banking in a technically coherent and operationally meaningful way. The prototype supports early-warning use cases, scenario testing, and decision-support workflows while also surfacing governance considerations such as fairness, explainability, privacy, reliability, and human oversight.

## 1. Introduction And Problem Statement

Retail banking was selected as the focus domain because financial distress is both economically important and operationally observable through sequential account behavior. Banks can often see salary timing, recurring obligations, spending pressure, payment performance, and credit utilization, but they do not always combine these signals in a way that captures the *path* into distress. Traditional scorecards and static rules are useful for broad segmentation, but they may miss the temporal progression that precedes a missed payment or overdraft event.

This project addresses the problem of **late detection**. In practice, a bank often reacts after a negative event has already taken place. A more proactive system should identify customers at risk of distress *before* a payment failure, overdraft, or persistent low-balance condition occurs. It should also allow the bank to compare different outreach or hardship-intervention strategies before acting.

The central idea is that customer financial behavior is naturally sequential. Salary credits affect repayment ability. Recurring bills reduce liquidity. Balance deterioration raises the likelihood of overdraft or failed payments. Support interventions can shift the path of future outcomes. These dependencies make foundation-model-style sequence learning particularly relevant to retail banking.

## 2. Innovation Thesis And MarS Alignment

The innovation thesis of BankSimFM is that a **financial foundation model for retail banking** can learn richer short-horizon risk dynamics than static summary models by operating directly on customer-event sequences. Instead of reducing a customer to aggregate features such as average balance or utilization, the model learns from the ordered interaction between incoming cash flow, recurring obligations, repayment stress, and account deterioration.

The project is inspired by **MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model**. MarS models financial markets as sequences of time-ordered events and uses a generative framework to support forecasting, detection, controllable simulation, and what-if analysis. BankSimFM adapts the same philosophy to retail banking. Rather than modeling order flow in a market, it models event flow in a customer's financial life.

[Insert Figure 1 here: MarS-to-BankSimFM Concept Map. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]

### 2.1 MarS To BankSimFM Mapping

| MarS concept | BankSimFM equivalent |
| --- | --- |
| Order sequence | Customer financial-event sequence |
| Market trajectory | Customer cash-flow and distress trajectory |
| Generative sequence model | Causal transformer over banking events |
| Clearing mechanism | Deterministic account-state engine |
| Controllable generation | Intervention-conditioned simulation |
| Market-risk detection | Financial distress early warning |
| What-if market analysis | Intervention scenario analysis |

BankSimFM is not a full reproduction of MarS. It is a simplified academic adaptation. The synthetic environment is much smaller, interventions are directional rather than causal, and the account-state engine remains the financial source of truth. Even so, the project preserves the most important conceptual elements: sequential modeling, controllable forecasting, path-dependent simulation, and analyst-facing interactivity.

## 3. Data Strategy And Preparation

The project uses a **hybrid data strategy**. For the report, it specifies the data that a real bank would need to train such a model. For the prototype, it uses synthetic customer-event data so the work is safe to share and demonstrate in a classroom environment.

### 3.1 Production Data Requirements

A real deployment would require four broad categories of data:

1. **Customer profile data**, such as income band, employment type, region, product holdings, tenure, and risk segment.
2. **Account and credit state data**, such as balance, overdraft limits, credit limits, utilization, and delinquency status.
3. **Event stream data**, such as salary credits, transfers, rent payments, utilities, card spend, loan due events, loan payments, missed payments, ATM withdrawals, fees, and support contacts.
4. **Outcome labels**, such as overdraft, missed payment, persistent low-balance flags, delinquency stages, and a 30-day distress label.

In production, the bank would collect these data from core banking systems, card ledgers, repayment systems, collections systems, and customer-service logs. Before training, the data would require:

- de-identification and access control
- timestamp normalization
- chronological ordering
- missing-value handling
- event taxonomy standardization
- customer-level train/validation/test splitting

### 3.2 Prototype Data Strategy

The implemented prototype uses **synthetic retail banking event sequences**. This choice was necessary because real banking data contain sensitive customer information and cannot be casually shared in a coursework repository. The synthetic environment preserves the structure of the problem while avoiding direct privacy risk.

The current generator simulates approximately `300` customers and `82,391` event rows under the default tuned configuration. It uses five archetypes:

- stable salaried
- volatile income
- high obligation
- rising utilization
- near distress

These archetypes are not fixed templates. The generator samples customer-specific traits such as income level, volatility, rent burden, spending intensity, repayment tendency, recovery tendency, starting buffer, and credit usage so that customers within the same archetype still behave differently.

The synthetic customer metadata also include fairness-ready segment fields:

- `income_band`
- `employment_type`
- `region`
- `risk_segment`

The prototype uses synthetic data only and contains **no direct customer identifiers and no PII**.

### 3.3 Event Schema And Labeling

Each event contains both event and state information. Key fields include:

| Field | Description |
| --- | --- |
| `customer_id` | Synthetic customer identifier |
| `event_timestamp` | Time of event |
| `event_type` | Event class |
| `amount` | Event amount |
| `amount_direction` | Credit or debit |
| `category` | Merchant or event category |
| `balance_before` | Account balance before event |
| `balance_after` | Account balance after event |
| `credit_limit` | Credit limit |
| `credit_utilization` | Utilization ratio |
| `days_to_next_due` | Days to next scheduled repayment |
| `intervention_flag` | Intervention marker |
| `distress_label_30d` | Distress within next 30 days |

For this project, distress is defined through future negative outcomes such as missed loan or credit-card payment, overdraft events, repeated low-balance stress, or other deterioration signals within the next 30 days.

### 3.4 Data Preprocessing And Training Sample Construction

The preprocessing pipeline sorts each customer's events strictly by time, derives sequence features, buckets selected continuous values, and constructs fixed-length history windows. The split is done at the **customer level** rather than the row level, which reduces leakage across train, validation, and test sets.

The current pipeline also carries internal due-state information so that restructuring-style interventions are visible to the model. Intervention-augmented windows are generated for the transformer to strengthen learned conditioning on simulated support actions.

An illustrative training sample structure is shown below.

| Component | Example |
| --- | --- |
| Context window | Last 256 customer events |
| Event tokens | `salary_credit -> rent_payment -> card_spend -> credit_card_payment_due -> card_spend` |
| Dense features | balance, utilization, days to next due, due-amount feature, intervention flag |
| Primary targets | next event, next amount bucket, next balance-delta bucket, distress label |
| Intervention token | `none`, `reminder`, `due_date_shift_7d`, `temporary_overdraft_buffer`, or `installment_restructure` |

## 4. Model Architecture And Training Approach

The project uses two sequence models:

- a **causal transformer** as the flagship model and current primary live scorer
- an **LSTM** as the benchmark baseline

[Insert Figure 2 here: BankSimFM System Architecture. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]  
[Insert Figure 3 here: Training And Evaluation Flow. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]

### 4.1 Why A Transformer

The transformer is well suited to this problem because customer distress emerges from interactions across time. A missed payment today may depend on salary timing, prior spending pressure, utilization buildup, and prior repayment burden. The transformer models these dependencies through a causal attention mechanism over ordered event histories.

### 4.2 Why Keep An LSTM Baseline

The LSTM remains important for two reasons. First, it provides a sequence-aware but simpler benchmark, which helps demonstrate whether the transformer's added complexity is justified. Second, comparison to a baseline is academically useful because it shows whether the MarS-inspired architecture actually improves performance.

### 4.3 Model Inputs And Objectives

The transformer consumes:

- event tokens
- dense account-state features
- intervention-aware state features
- internal due-state information

The current multitask transformer is trained to predict:

- next event
- next amount bucket
- next balance-delta bucket
- 30-day distress probability

The LSTM is trained as a benchmark classifier on the same distress target.

### 4.4 Training Process

Training uses:

- customer-level train/validation/test splits
- class weighting for distress
- validation-based threshold selection
- early stopping and checkpoint saving
- verbose epoch-level logging

The current tuned defaults include:

- transformer learning rate `5e-4`
- LSTM learning rate `1e-3`
- max epochs `12`
- patience `5`
- intervention augmentation rate `0.30`
- intervention augmentation steps `3`
- transformer distress loss weight `1.5`

### 4.5 Infrastructure Assumptions

The system is designed to run on commodity development hardware. Apple Silicon support is implemented through the `mps` device path when available, with `cuda` and `cpu` as fallbacks. This is appropriate for a course-project prototype, although a production-scale foundation-model program would require more compute, more data, and stronger experiment management.

## 5. Forecasting, Scoring, And Intervention Simulation

The prototype exposes three public interfaces:

- `score_customer(history, horizon_days=30)`
- `forecast_customer(history, horizon_days=30)`
- `simulate_intervention(history, intervention_type, horizon_days=30)`

[Insert Figure 4 here: Baseline And Intervention Simulation Flow. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]

### 5.1 Live Scoring

The transformer is now the **primary live scorer**. This means that current observed customer histories are scored with the transformer distress head and the saved transformer threshold. The LSTM remains available as the benchmark model for comparison in metrics and governance views.

### 5.2 Forecasting

Forecasting is model-driven rather than purely heuristic. The transformer performs autoregressive decoding to produce likely short-horizon event trajectories. A deterministic account-state engine then updates balances and utilization so that decoded paths remain financially coherent.

### 5.3 Intervention Simulation

The system supports four in-scope interventions:

- `reminder`
- `due_date_shift_7d`
- `temporary_overdraft_buffer`
- `installment_restructure`

Intervention simulation works by adjusting the internal account state, decoding an intervention-conditioned future path, and then comparing the resulting scenario to the baseline path. Scenario-risk rescoring for simulated futures is now transformer-based, which keeps the simulation logic aligned with the forecasting model.

This design is important because it makes the what-if output reflect the same sequence model that generated the scenario. However, the system should still be interpreted as a directional simulator, not a causal estimator of real-world intervention effect.

## 6. Evaluation And Results

The project evaluates both models on the same 30-day distress task and also generates supporting artifacts for early warning, fairness, simulation realism, intervention usefulness, and repeated-run stability.

### 6.1 Holdout Classification Results

| Model | Test AUC | Test Precision | Test Recall | Test F1 | Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | `0.8342` | `0.6127` | `0.5556` | `0.5828` | `0.8696` |
| LSTM | `0.8126` | `0.6458` | `0.4250` | `0.5126` | `0.8675` |

These results justify the switch to the transformer as the primary live scorer. The transformer outperforms the LSTM on the most important composite indicators in this prototype, especially AUC and F1, while maintaining reasonable precision and recall balance.

### 6.2 Early-Warning And Scenario Metrics

The current simulation artifact reports the following transformer early-warning summary:

- average lead time before distress events: `21.5455` days
- hit rate on distressed customers: `0.7333`
- false positive rate on stable customers: `0.4333`

Simulation quality and stability metrics are also available:

- event-mix divergence: `0.675`
- balance-trajectory RMSE: `371.8487`
- utilization RMSE: `0.0282`
- ending-balance standard deviation across repeat runs: `75.4332`
- negative-event-count standard deviation: `0.257`

These values show that the prototype is capable of generating non-trivial scenario variation while staying anchored to a deterministic financial-state engine.

### 6.3 Intervention Usefulness

The current average predicted 30-day risk reduction by intervention is:

| Intervention | Avg Predicted Risk Reduction | Share Improved By 5pp |
| --- | ---: | ---: |
| `due_date_shift_7d` | `0.0171` | `0.35` |
| `installment_restructure` | `0.0228` | `0.40` |
| `temporary_overdraft_buffer` | `0.0217` | `0.35` |
| `reminder` | `-0.0447` | `0.35` |

This is an important result. The negative average result for `reminder` should **not** be interpreted as a code bug. Instead, it shows that not all interventions are equally helpful in the current synthetic environment. Stronger structural interventions such as installment restructuring or temporary liquidity support can outperform lightweight reminder-based outreach.

### 6.4 Fairness And Segment Behavior

The project also computes subgroup metrics across:

- archetype
- income band
- employment type
- region
- risk segment

Subgroup performance is informative but uneven. For example, the transformer performs strongly on several higher-risk groups, including `high_obligation` (AUC `0.9175`) and `rising_utilization` (AUC `0.9105`). At the same time, performance varies across regions and employment types, which means subgroup monitoring would remain necessary in any serious deployment path.

The report should interpret these fairness results cautiously. They are useful for governance discussion, but they are still shaped by a synthetic data generator rather than real customer populations.

## 7. Downstream Applications And Business Value

The trained prototype supports four main downstream applications.

[Insert Figure 5 here: Analyst Workflow And Collections Prioritization. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]

### 7.1 Early Distress Warning

The bank can use the live scoring interface to identify customers likely to enter distress within the next 30 days. This allows earlier review, targeted outreach, and more proactive servicing before a failure event appears in the ledger.

### 7.2 Intervention What-If Analysis

Analysts can compare baseline and intervention-conditioned trajectories to estimate whether a due-date shift, restructuring action, or temporary liquidity support is likely to reduce short-horizon distress risk. This helps prioritize actions more intelligently than treating all customers the same.

### 7.3 Portfolio Stress Monitoring

The dashboard includes cohort-level monitoring by archetype and segment fields such as income band, employment type, region, and risk segment. This allows the bank to see where distress risk is building and which segments are more sensitive to simulated support policies.

### 7.4 Collections Prioritization

The Collections Prioritization page ranks customers by actionable risk and recommended intervention. This is a particularly practical downstream workflow because it connects model outputs to an analyst's daily queue. Instead of simply flagging risk, the prototype suggests which customer to review first and which intervention appears most promising under the current scenario logic.

### 7.5 Translation Into Operational And Economic Decisions

| Application | Operational decision | Business value mechanism |
| --- | --- | --- |
| Early warning | Review or contact customers earlier | Avoid missed-payment losses and downstream delinquency costs |
| What-if simulation | Select best support action | Improve intervention targeting and reduce wasted outreach |
| Portfolio monitoring | Watch vulnerable cohorts | Improve segment-level planning and staffing |
| Collections prioritization | Rank customers for action | Increase analyst productivity and intervention efficiency |

## 8. Governance, Risks, And Controls

[Insert Figure 6 here: Governance And Monitoring Control Loop. See [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).]

The project surfaces several important governance themes.

### 8.1 Privacy

The current prototype uses synthetic data only, which reduces privacy risk materially. No real PII is stored in the repository or required for the classroom demo. In a production bank setting, strict controls would still be needed around data access, lineage, encryption, retention, and model-development environments.

### 8.2 Fairness

Fairness is addressed by including segment metadata and computing subgroup metrics across several segment dimensions. The results show that subgroup behavior is not uniform. This is a useful governance signal, but it should not be overinterpreted because the subgroup structure is synthetic and therefore does not guarantee real-world representativeness.

### 8.3 Explainability

The system supports partial explainability through:

- recent risk-signal summaries
- top-driver heuristics
- timeline review in the dashboard
- explicit baseline versus intervention comparison

This is not full causal explanation, but it gives analysts more context than a raw distress score alone.

### 8.4 Reliability And Operational Risk

The deterministic account-state engine improves simulation consistency by grounding decoded futures in explicit balance and utilization updates. Even so, the system remains a prototype and should not be treated as a production-grade decision engine. Operational risks include distribution shift, calibration drift, subgroup instability, and misuse of intervention outputs as if they were causal evidence.

### 8.5 Controls

If a real bank were to pursue this idea, the following controls would be necessary:

- human-in-the-loop review for high-impact actions
- ongoing fairness monitoring
- periodic retraining and threshold review
- monitoring for drift and intervention instability
- strict separation between advisory analytics and autonomous adverse-action decisions

This prototype should **not** be used for live autonomous credit decisions.

## 9. Economic Value Measurement Methodology

The project does not claim realized economic value. Instead, it proposes a methodology for estimating value under scenario assumptions.

### 9.1 Value Drivers

Potential value drivers include:

- avoided loss from missed payments and early delinquency
- reduced collections effort on low-yield cases
- better targeting of hardship or support interventions
- analyst productivity gains from ranked triage workflows
- customer retention uplift from timely and appropriate support

### 9.2 Scenario-Based Measurement Framework

| Use case | Decision lever | Value driver | Example measurement |
| --- | --- | --- | --- |
| Early warning | Trigger earlier review | Lower missed-payment loss | Reduction in missed-payment incidence among contacted cases |
| What-if simulation | Choose more effective intervention | Better intervention ROI | Average risk reduction per intervention type |
| Portfolio monitoring | Shift staffing or policy focus | Lower operational surprises | Distress trend by segment and action rate |
| Collections prioritization | Rank customers for outreach | Higher analyst efficiency | Cases reviewed per analyst hour and avoided downstream loss |

In a real implementation, the bank would estimate value by combining historical base rates, intervention costs, analyst capacity, and outcome deltas. The current project provides the decision-support structure, not a realized business case.

## 10. Assumptions, Limitations, And Future Work

This report rests on several important assumptions:

- the prototype uses synthetic data only
- the default distress horizon is 30 days
- the environment is prototype-scale rather than production-scale
- intervention outputs are directional and scenario-based, not causal proof

The project also has meaningful limitations:

- the synthetic world remains simpler than real retail banking behavior
- fairness results depend on the synthetic segment design
- the intervention simulator is informative but not validated on real customer outcomes
- the work does not yet include a multi-seed stability study
- the dashboard and artifacts are designed for coursework demonstration rather than enterprise deployment

Recommended future work includes:

- training and validation on real or more richly calibrated data
- formal calibration analysis
- repeated-seed stability testing
- additional intervention types
- longer-horizon forecasting studies
- deeper economic-value modeling with explicit cost assumptions

## 11. Conclusion

BankSimFM demonstrates that the core ideas behind MarS can be adapted from market-order simulation to retail banking customer-event simulation. The prototype shows that a sequence-based financial foundation model can support short-horizon distress warning, future-path forecasting, intervention what-if analysis, portfolio monitoring, and collections prioritization within a single coherent system.

The project also demonstrates technical understanding across data design, sequence modeling, simulation, downstream applications, governance, and economic framing. Most importantly, the latest results show that the transformer is now the strongest live model in the implemented system, which strengthens the MarS-inspired design choice and gives the prototype a credible empirical story for the MH6818 project.

## Appendix A: Current Metric Snapshot

### Holdout Metrics

| Model | Validation AUC | Test AUC | Test F1 |
| --- | ---: | ---: | ---: |
| Transformer | `0.8581` | `0.8342` | `0.5828` |
| LSTM | `0.7969` | `0.8126` | `0.5126` |

### Notable Simulation Findings

- transformer early-warning lead time is approximately `21.5` days
- `installment_restructure` has the strongest positive average risk reduction among tested interventions
- `reminder` is negative on average and should be interpreted as a weaker intervention in the current synthetic environment

## Appendix B: Figures To Insert

- Figure 1. MarS-to-BankSimFM Concept Map
- Figure 2. BankSimFM System Architecture
- Figure 3. Training And Evaluation Flow
- Figure 4. Baseline And Intervention Simulation Flow
- Figure 5. Analyst Workflow And Collections Prioritization
- Figure 6. Governance And Monitoring Control Loop

All diagram sources are stored in [docs/architecture.md](/Users/abhishek/Desktop/Projects/regtech/docs/architecture.md).
