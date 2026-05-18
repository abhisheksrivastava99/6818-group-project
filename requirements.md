# BankSimFM: Requirements Specification

## 1. Project Overview

**BankSimFM** is a retail banking financial distress early-warning simulator inspired by **MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model**. The project adapts MarS from market-order simulation to **customer financial-event simulation**. Instead of modeling order flow in a limit order book, BankSimFM models time-ordered retail banking events such as salary credits, bill payments, card spending, EMI due events, EMI payments, overdraft usage, and support interactions.

The system is designed to learn sequential customer financial behavior, estimate the probability of financial distress over the next 30 days, forecast likely future account trajectories, and simulate the effect of support interventions such as reminders or due-date shifts. This is a **course-project prototype** aligned to the MH6818 group project brief, not a production banking platform.

## 2. Alignment With MH6818 Project Requirements

This specification is designed to satisfy the project brief in [MH6818_Group_Project.md](/Users/abhishek/Desktop/Projects/regtech/MH6818_Group_Project.md):

1. **Selected domain:** Retail banking
2. **Innovation thesis:** Foundation models can improve early warning, forecasting, and intervention planning by learning customer event sequences rather than static tabular summaries
3. **Data strategy:** Define production data requirements and build a prototype on synthetic event sequences
4. **Model architecture:** Use a MarS-inspired causal transformer as the flagship model and LSTM as the main sequence baseline
5. **Downstream applications:** Provide multiple retail banking use cases enabled by the model
6. **Governance:** Address privacy, fairness, explainability, reliability, and operational risk
7. **Economic value:** Define a measurable value-creation framework
8. **Assumptions:** State all modeling and deployment assumptions explicitly

## 3. Problem Statement And Business Motivation

Retail banks often detect customer distress only after a negative event has already occurred, such as a missed payment, overdraft, or failed recurring debit. Traditional rules and point-in-time scorecards may capture basic risk patterns, but they often miss **temporal dependencies** across salary timing, recurring obligations, rising utilization, and unstable spending.

The bank needs a system that can:

- identify customers likely to enter distress in the near future
- forecast the likely path of balances, repayments, and negative events
- test potential interventions before acting
- support earlier and more targeted outreach

### Target User For MVP

The primary user for v1 is a:

- risk analyst
- collections analyst
- customer support or hardship operations analyst

### Core Business Questions

- Which customers are likely to experience financial distress in the next 30 days?
- What events are likely to happen next in the customer’s financial timeline?
- Which intervention is most likely to reduce the customer’s short-term distress risk?

## 4. MarS Inspiration And Domain Adaptation

MarS is built around high-resolution sequential market modeling, controllable generation, and interactivity. BankSimFM preserves these ideas and maps them into retail banking.

### MarS To BankSimFM Mapping

| MarS concept | BankSimFM analogue |
| --- | --- |
| Order sequence | Customer financial-event sequence |
| Market trajectory | Customer cash-flow and distress trajectory |
| Causal transformer | Causal event transformer over banking events |
| Simulated clearing house | Deterministic account-state engine |
| Controllable generation | Intervention-conditioned simulation |
| Detection system | Early distress and anomaly warning |
| What-if market analysis | What-if customer intervention analysis |

### Why A Causal Sequence Model

MarS uses a **causal transformer** because future market events depend on prior realized events. Retail banking has the same causal structure:

- salary credits affect the ability to pay bills
- recurring bills reduce available liquidity
- balance stress increases the chance of overdraft or missed payment
- support actions may alter future customer outcomes

This makes a **decoder-only causal event transformer** the most natural MarS-inspired architecture for the project.

## 5. Product Scope

### MVP Objective

The MVP will provide a synthetic-data prototype that:

- ingests a customer’s ordered financial event history
- outputs a **30-day distress risk score**
- forecasts likely next events and account trajectory
- compares baseline versus intervention scenarios
- highlights the main behavioral drivers of predicted distress

### Definition Of Financial Distress For V1

For this project, a customer is considered to be in financial distress if one or more of the following occurs within the prediction horizon:

- missed loan or credit card payment
- overdraft event
- repeated low-balance condition
- credit utilization above threshold
- repeated failed debit or payment attempts

### In-Scope Interventions

The v1 simulator will support these intervention scenarios:

- `reminder`
- `due_date_shift_7d`
- `temporary_overdraft_buffer`
- `installment_restructure`

### Non-Goals For V1

- no real customer PII
- no live integration with banking systems
- no autonomous production decisioning
- no customer-facing mobile application
- no hard guarantee of causal impact in the prototype

## 6. Functional Requirements

### Core Functional Capabilities

The system must:

1. accept a chronological customer event history
2. compute a distress risk score for a configurable horizon, defaulting to 30 days
3. forecast a likely short-horizon event and balance trajectory
4. simulate intervention-conditioned future paths
5. compare baseline and intervention outcomes
6. expose results through a Streamlit analyst demo

### Prototype Interfaces

The requirements document defines the following logical interfaces for the prototype:

```python
score_customer(history, horizon_days=30)
```

Returns:

- distress probability
- distress label
- top contributing drivers
- summary of recent risk signals

```python
forecast_customer(history, horizon_days)
```

Returns:

- projected sequence of future events
- projected balance path
- projected utilization path
- forecast summary

```python
simulate_intervention(history, intervention_type, horizon_days)
```

Returns:

- baseline risk and trajectory
- intervention risk and trajectory
- delta in risk
- key scenario differences

## 7. Data Requirements

The project will use a **hybrid data strategy**:

- **production framing:** define the real data required by a bank
- **prototype implementation:** use synthetic customer-event data

### 7.1 Production Data Requirements

#### Customer Profile Data

- customer tenure
- age band
- income band
- employment type
- region
- product holdings
- risk segment

#### Account And Credit State

- deposit account balance
- average monthly balance
- overdraft limit
- credit card limit
- current credit utilization
- delinquency status

#### Event Stream Data

- salary credit
- transfer in
- transfer out
- rent payment
- utility payment
- grocery spend
- transportation spend
- loan EMI due
- loan EMI paid
- loan EMI missed
- credit card payment due
- credit card payment made
- card spend
- ATM withdrawal
- bank fee or penalty
- support contact or hardship request

#### Labels

- missed payment
- overdraft
- persistent low-balance flag
- delinquency stage
- 30-day distress label

### 7.2 Prototype Data Requirements

The prototype will use **synthetic customer-event sequences** designed to resemble common retail banking patterns:

- stable salaried customers
- volatile income customers
- high recurring obligation customers
- rising utilization customers
- near-distress and distressed customers

Optional calibration can be done using public household cash-flow distributions if suitable data is found, but the prototype does not depend on real data.

All prototype and demo data must be:

- synthetic or fully anonymized
- free of direct identifiers
- safe to share in a classroom demo or public repo

### 7.3 Core Event Schema

Each event record should contain the following fields:

| Field | Description |
| --- | --- |
| `customer_id` | Synthetic customer identifier |
| `event_timestamp` | Event time |
| `event_type` | Event class |
| `amount` | Event amount |
| `amount_direction` | Credit or debit |
| `category` | Merchant or event category |
| `balance_before` | Account balance before event |
| `balance_after` | Account balance after event |
| `credit_limit` | Available credit limit |
| `credit_utilization` | Credit utilization ratio |
| `days_to_next_due` | Days to next scheduled repayment |
| `intervention_flag` | Intervention marker if present |
| `distress_label_30d` | Binary label for distress within 30 days |

### 7.4 Data Preprocessing Requirements

The data pipeline must:

- sort all events strictly by time
- split train, validation, and test data by **customer**, not by event
- encode event types categorically
- bucket continuous values such as amount, balance, and utilization for generative modeling
- construct trailing histories up to **256 events**
- exclude free-text sensitive information
- handle missing values with deterministic defaults or explicit missing tokens

### 7.5 Training Sample Construction

Each training sample will be a customer history window:

- input: ordered event sequence up to time `t`
- target 1: next event attributes
- target 2: 30-day distress outcome
- optional control: intervention token

This allows one shared sequence backbone to support both **forecasting** and **distress detection**.

## 8. Model Strategy

### 8.1 Primary Model: Decoder-Only Causal Transformer

The primary model is a **decoder-only causal transformer** modeled after MarS’s order-level sequence modeling idea.

#### Role

- flagship MarS-inspired foundation model
- sequence forecasting model
- controllable simulator
- distress classification model with auxiliary head

#### Prototype Architecture

- 4 transformer layers
- hidden size: `256`
- attention heads: `8`
- maximum context window: `256` events
- dropout: configurable, default `0.1`

#### Input Representation

Each event token will combine embeddings for:

- event type
- amount bucket
- balance bucket
- utilization bucket
- time-gap bucket
- intervention token
- positional order

#### Primary Tasks

- next-event prediction
- next amount bucket prediction
- next balance-delta bucket prediction
- 30-day distress prediction
- counterfactual simulation under interventions

#### Why This Model Is The Main Choice

- strongest alignment with MarS
- preserves sequential causality
- supports autoregressive generation of future events
- supports intervention-conditioned what-if analysis
- fits the “financial foundation model” framing required by the assignment

### 8.2 Baseline Model: LSTM

The baseline model is a **Long Short-Term Memory (LSTM)** network over customer event sequences.

#### Role

- benchmark sequential model
- simpler alternative to the transformer
- reference point for distress prediction accuracy

#### Expected Use

- consume ordered event features or embedded event sequences
- output 30-day distress probability
- compare classification performance against the transformer

#### Why LSTM Is Included

- sequence-native baseline
- easier to train on a small synthetic dataset
- academically cleaner than comparing against a non-sequential tabular model

### 8.3 Optional Stretch Model

If time permits, the team may include a **GRU** benchmark. This is optional and not required for MVP.

## 9. Training And Simulation Design

### 9.1 Transformer Training Objective

The transformer training setup should include:

- **autoregressive next-event objective** over the ordered event stream
- **auxiliary distress classification head** for 30-day risk
- intervention token conditioning for counterfactual generation

Loss can be defined as a weighted combination of:

- next-event cross-entropy
- amount/balance bucket prediction loss
- distress classification loss

### 9.2 LSTM Training Objective

The LSTM should focus on:

- input: ordered customer event history
- output: 30-day distress probability
- loss: binary cross-entropy or weighted binary loss if classes are imbalanced

### 9.3 Deterministic Account-State Engine

MarS uses a simulated clearing house to make generated sequences interact with realized outcomes. BankSimFM will use a **deterministic account-state engine** that updates:

- balance
- due status
- utilization
- overdraft status
- payment-missed indicators

This engine is required so that simulated events remain financially consistent.

### 9.4 Simulation Workflow

The scenario engine should:

1. start from observed customer history
2. condition on no intervention or a selected intervention
3. generate future events step by step
4. update account state after each generated event
5. aggregate the resulting risk and projected path
6. compare baseline versus intervention outcomes

### 9.5 Training Process

The training plan should specify:

- train/validation/test split by customer
- early stopping on validation performance
- model checkpointing
- class-balance monitoring
- multiple random seeds if time permits

### 9.6 Infrastructure Assumptions

For the course-project prototype:

- development on laptop or lab machine is acceptable
- small-GPU training is preferred if available
- CPU-only training is acceptable for the LSTM baseline and small-scale transformer experiments
- data volume is expected to be small because of the synthetic prototype setup

## 10. Evaluation Plan

### 10.1 Classification Metrics

- AUC
- precision
- recall
- F1

### 10.2 Early-Warning Metrics

- average lead time before missed payment or overdraft
- hit rate on distressed customers
- false positive rate for stable customers

### 10.3 Simulation Quality Metrics

- plausibility of generated event mix
- realism of balance trajectories
- similarity of simulated distress rates to synthetic generator assumptions
- stability of results under repeated simulation runs

### 10.4 Intervention Usefulness Metrics

- reduction in predicted distress under intervention
- proportion of customers whose forecast changes materially
- comparison of intervention scenarios for the same customer

### 10.5 Benchmarking Requirement

The final report should compare:

- transformer versus LSTM
- on the same holdout split
- using the same 30-day distress target

## 11. Downstream Applications

The trained BankSimFM model should support the following retail banking applications.

### 11.1 Early Distress Warning

Detect customers likely to miss payments or enter overdraft before the event occurs.

### 11.2 Intervention Testing

Evaluate whether reminders, due-date shifts, or temporary relief reduce predicted distress.

### 11.3 Collections Prioritization

Help collections teams prioritize customers based on projected risk and likely benefit from outreach.

### 11.4 Portfolio Stress Monitoring

Aggregate customer-level simulations to identify segments with elevated distress risk under different assumptions.

### 11.5 Synthetic Scenario Generation

Generate realistic customer-event sequences for safe sandbox testing and model development without real customer PII.

## 12. Streamlit Frontend Requirements

The frontend will be a **presentation-first analyst demo** built in Streamlit.

### 12.1 Frontend Goals

- explain the project clearly to instructors
- demonstrate the value of sequential modeling and simulation
- allow interactive review of customer timelines and what-if outcomes

### 12.2 Required Pages

#### Overview

Must show:

- project summary
- MarS inspiration summary
- KPI cards
- synthetic dataset summary
- cohort-level distress distribution

#### Customer Explorer

Must show:

- customer selector
- event timeline
- balance chart
- recent event table
- current distress score
- key risk drivers

#### What-If Simulator

Must show:

- intervention selector
- horizon selector: 30, 60, or 90 days
- baseline versus intervention risk comparison
- projected balance trajectory comparison
- top forecasted negative events

#### Model And Governance

Must show:

- model architecture summary
- transformer versus LSTM rationale
- synthetic-data disclaimer
- fairness, privacy, explainability, and reliability notes

### 12.3 Required UI Components

- sidebar filters
- metric cards
- line charts
- tables for event history
- scenario comparison panel
- visible “synthetic demo data” label

### 12.4 Frontend Non-Functional Requirements

- load quickly with packaged sample data
- require no secrets for MVP
- remain usable on laptop display during presentation
- support reproducible demo flow

## 13. Deployment Requirements

### Primary Deployment Mode

- local Streamlit run for development and presentation rehearsal

### Optional Deployment Mode

- Streamlit Community Cloud for a shareable demo, only if the app uses synthetic data and lightweight artifacts

### Deployment Constraints

- no real customer data
- no secrets required for core demo
- no external database required for MVP
- static packaged sample data preferred
- lightweight model artifacts or cached predictions preferred

## 14. Governance, Risk, And Controls

### Privacy

- use only synthetic or anonymized data in the prototype
- document data minimization principles for production use

### Fairness

- compare model performance across segments such as income band or employment type
- watch for systematic over-flagging of vulnerable groups

### Explainability

- show timeline-based risk drivers
- expose understandable scenario outputs rather than only raw scores

### Reliability

- monitor for drift in real deployment scenarios
- require human review before customer action in a production setting

### Operational Risk

- false positives can cause unnecessary outreach and customer friction
- false negatives can miss customers who need early support

### Model Limitations

- synthetic data may not capture all real-world distress patterns
- simulated intervention impact is directional, not causal proof
- prototype results should not be treated as live credit decisions

## 15. Economic Value Measurement

The project should propose a methodology, not claim realized bank value.

### Value Drivers

- reduced losses from missed payments and overdrafts
- lower collections cost through better targeting
- improved customer retention through earlier assistance
- operational savings from better scenario prioritization

### Measurement Approach

Estimate value using scenario analysis:

- number of high-risk customers identified earlier
- assumed intervention success rate
- avoided loss per prevented missed payment or overdraft
- reduced manual outreach cost
- uplift in retention for supported customers

This section must make clear that the prototype demonstrates **how** value would be measured, not that the value has already been realized.

## 16. Assumptions

- this is a balanced MVP for a university group project
- the prototype uses synthetic data while documenting real production data needs
- the main prediction horizon is 30 days
- the causal transformer is the flagship model because of MarS alignment
- the LSTM is the required benchmark baseline
- Streamlit is optimized for demo clarity, not enterprise workflow depth
- deployment is optional and only appropriate for synthetic-only artifacts

## 17. Deliverables Enabled By This Specification

This requirements specification is intended to guide the creation of:

- final report sections
- PowerPoint presentation content
- sample synthetic dataset
- prototype modeling code
- Streamlit demo frontend

## 18. Acceptance Criteria

`requirements.md` is complete when it:

- clearly explains the retail banking use case
- directly connects the design to the MarS paper
- identifies required production data and prototype synthetic data
- defines a usable event schema and preprocessing rules
- specifies the causal transformer as the main model and LSTM as the baseline
- defines the simulation engine and intervention logic
- lists downstream applications, governance requirements, and economic value methodology
- specifies the Streamlit frontend and deployment approach
- is detailed enough for implementation without major product-level ambiguity
