# BankSimFM Implementation And MarS Alignment Guide

This document explains what has been implemented in the BankSimFM project so far, why it was built this way, how it aligns with the MarS paper, and what problem the prototype is solving. It is written as a study guide so that a team member can understand the project quickly before writing the report, preparing slides, or giving the presentation.

## 1. Problem Statement

Retail banks want to identify customers who may enter short-term financial distress before the negative event actually happens.

In practice, banks often react after visible damage has already occurred:

- a missed EMI payment
- a failed credit-card debit
- an overdraft
- persistent low balances
- rapidly rising credit utilization

Traditional rules or static scorecards can capture some risk, but they often miss the sequence of events that leads to distress. For example:

- salary comes in late
- rent and utilities hit first
- card spending grows
- due payment arrives when balance is already stressed
- repayment fails or is missed

This project models that evolving timeline directly instead of using only static summary variables.

## 2. What The Prototype Is Trying To Achieve

The BankSimFM prototype has four main goals:

1. Learn patterns from ordered customer financial-event histories.
2. Estimate the probability that a customer will experience distress in the next 30 days.
3. Forecast likely short-horizon customer events and balance behavior.
4. Simulate how interventions may change that future path.

This is not a production credit system. It is a course-project prototype designed to demonstrate how a MarS-inspired financial foundation model idea can be adapted from markets to retail banking.

## 3. High-Level Idea

The MarS paper models a financial market as a sequence of time-ordered events and uses a generative foundation model to simulate future market trajectories.

BankSimFM applies the same idea to retail banking:

- MarS models order sequences in a market.
- BankSimFM models event sequences in a customer’s financial life.

Instead of orders, the sequence contains events such as:

- `salary_credit`
- `rent_payment`
- `utility_payment`
- `loan_emi_due`
- `loan_emi_paid`
- `loan_emi_missed`
- `credit_card_payment_due`
- `credit_card_payment_made`
- `card_spend`
- `failed_debit`
- `overdraft_event`
- `support_contact`

The model then tries to understand how these events interact over time and whether they signal future distress.

## 4. How This Aligns With The MarS Paper

## 4.1 Core MarS Concepts

The MarS paper introduces a financial market simulation engine built around:

- high-resolution sequential modeling
- controllable generation
- interactivity with a clearing mechanism
- forecasting
- detection
- what-if analysis

## 4.2 Our Mapping From MarS To BankSimFM

| MarS Concept | BankSimFM Equivalent |
| --- | --- |
| Order sequence | Customer financial-event sequence |
| Market trajectory | Customer cash-flow and distress trajectory |
| Large Market Model | Customer event-sequence model |
| Simulated clearing house | Deterministic account-state engine |
| Forecasting future market behavior | Forecasting future customer events |
| Market-risk detection | Financial distress detection |
| What-if market analysis | Intervention simulation |

## 4.3 Where We Align Well

The current implementation aligns well with MarS in these ways:

- it treats the problem as a sequence problem rather than a static tabular problem
- it uses a transformer as the flagship model
- it includes forecasting, risk detection, and what-if analysis
- it uses an account-state engine to keep simulations financially consistent
- it frames simulation as path-dependent over time

## 4.4 Where We Are A Simplified Prototype

The current project is still a simplified academic version of the full MarS vision:

- intervention effects remain directional within a synthetic simulator rather than causal proof for real customers
- the deterministic account-state engine still acts as the financial source of truth, so generation is constrained rather than free-form
- the synthetic world is much smaller and more structured than the large-scale market environment imagined by MarS
- the simulator is customer-level and lightweight, not a large-scale world model

So the project is strongly inspired by MarS, but it is not a one-to-one reproduction. It is an adaptation of the MarS philosophy to a new financial domain.

## 5. What We Implemented

The project currently includes five major implementation blocks:

1. Synthetic data generation
2. Data preprocessing and training-sample construction
3. Sequence models
4. Inference and scenario simulation
5. Streamlit demo application

## 6. Architecture Overview

The system can be understood as the following pipeline:

1. Generate synthetic customer-event histories.
2. Convert those histories into model-ready windows.
3. Train sequence models on the windows.
4. Use trained models to score customer distress and decode future trajectories.
5. Use learned intervention-conditioned decoding plus the state engine to simulate scenarios.
6. Present results in a Streamlit dashboard.

### Architecture Flow

```text
Synthetic Customer Profiles
        ->
Synthetic Event Generator
        ->
Chronological Event Table
        ->
Customer Metadata And Segment Labels
        ->
Feature Encoding + Bucketing + Customer Split
        ->
Customer History Windows + Intervention-Augmented Windows
        ->
Transformer + LSTM Training
        ->
Saved Artifacts, Metrics, Fairness, And Simulation Summaries
        ->
Inference APIs
        ->
Streamlit Dashboard
```

## 7. Data Design And What We Generated

## 7.1 Why Synthetic Data

Real banking data is sensitive and cannot be casually shared for a class project. The project therefore uses synthetic data so that:

- there is no customer PII
- the repository is safe to share
- the demo is reproducible
- the prototype still reflects realistic event sequences

## 7.2 Customer Archetypes

The generator creates five customer archetypes in [src/banksimfm/data/generator.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/generator.py):

- stable salaried
- volatile income
- high obligation
- rising utilization
- near distress

Each archetype has different behavioral tendencies such as:

- income range
- spending volatility
- rent ratio
- repayment tendency
- recovery tendency
- credit usage behavior

These settings allow the project to create more realistic variation than a simple random generator.

## 7.3 Event Schema

The event schema is defined in [src/banksimfm/data/schema.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/schema.py).

Each event includes fields such as:

- `customer_id`
- `event_timestamp`
- `event_type`
- `amount`
- `amount_direction`
- `category`
- `balance_before`
- `balance_after`
- `credit_limit`
- `credit_utilization`
- `days_to_next_due`
- `intervention_flag`
- `distress_label_30d`

This schema is important because it captures both:

- the event itself
- the account state after the event

That makes the sequence meaningful for both learning and simulation.

## 7.4 Distress Labels

The project labels each event with whether distress happens within the next 30 days. This forward-looking label is created in [src/banksimfm/data/generator.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/generator.py).

Distress events currently include:

- `loan_emi_missed`
- `failed_debit`
- `overdraft_event`

This means the model is not just learning “is this event bad now?” It is learning “does this history imply trouble soon?”

## 8. Data Pipeline And Training Samples

The preprocessing pipeline is implemented in [src/banksimfm/data/pipeline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/data/pipeline.py).

## 8.1 What The Pipeline Does

It:

- sorts events chronologically
- encodes event types as IDs
- buckets amount, balance, utilization, and time gaps
- computes simple dense features
- splits customers into train, validation, and test sets
- builds trailing history windows for training

## 8.2 Why Customer-Level Splitting Matters

We split by customer instead of by event so that the model is tested on unseen customers. This is important because if the same customer appears in both train and test, results can be misleadingly optimistic.

## 8.3 How A Training Sample Looks

For each customer history window:

- input = past ordered events up to time `t`
- target 1 = next event type
- target 2 = next amount bucket
- target 3 = next balance-delta bucket
- target 4 = whether distress occurs within 30 days

This supports the idea that one shared sequence backbone can help with both forecasting and risk detection.

## 9. Models

## 9.1 Transformer Model

The transformer is defined in [src/banksimfm/models/transformer.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/transformer.py).

### Inputs Used

The transformer uses embeddings for:

- event type
- amount bucket
- balance bucket
- utilization bucket
- time-gap bucket
- intervention token
- position

These embeddings are summed to form the event representation at each time step.

### Outputs Produced

The current transformer produces:

- next-event logits
- next-amount-bucket logits
- next-balance-delta-bucket logits
- distress logit

So it is doing two jobs:

- sequence understanding for what event, amount regime, and balance movement come next
- classification for whether the customer is at short-term distress risk

### Why This Matters

This is the part of the project that is most aligned with the “foundation model” idea from MarS. It treats event histories as a structured sequence and trains a single backbone to support multiple tasks.

## 9.2 LSTM Baseline

The LSTM baseline is defined in [src/banksimfm/models/baseline.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/baseline.py).

It consumes dense sequential features and outputs a distress logit.

The LSTM is important because:

- it is a sequence-native baseline
- it is simpler than the transformer
- it provides a benchmark so we can compare whether the more advanced architecture is justified

## 10. Training Process

Training is implemented in [src/banksimfm/models/training.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/models/training.py).

## 10.1 What Is Trained

Two models are trained:

- transformer
- LSTM

## 10.2 Training Objectives

Current training includes:

- cross-entropy loss for next-event prediction in the transformer
- cross-entropy loss for next amount-bucket prediction in the transformer
- cross-entropy loss for next balance-delta-bucket prediction in the transformer
- binary cross-entropy loss for 30-day distress prediction
- intervention-conditioned augmented training windows for the transformer

Class imbalance is handled using a positive class weight for distress prediction.

## 10.3 Validation And Testing

The pipeline:

- trains on training windows
- tracks validation performance
- uses early stopping patience
- saves the best checkpoints
- evaluates the saved models on the holdout test set

## 10.4 Metrics

The current saved metrics include:

- AUC
- precision
- recall
- F1
- accuracy
- selected classification threshold

These are stored in [artifacts/metrics.json](/Users/abhishek/Desktop/Projects/regtech/artifacts/metrics.json).

Additional evaluation artifacts now include:

- [artifacts/simulation_metrics.json](/Users/abhishek/Desktop/Projects/regtech/artifacts/simulation_metrics.json)
- [artifacts/fairness_metrics.json](/Users/abhishek/Desktop/Projects/regtech/artifacts/fairness_metrics.json)

These cover:

- early-warning behavior
- simulation realism
- repeated-run stability
- intervention usefulness
- subgroup fairness breakdowns

## 11. Inference And Public APIs

The public APIs are implemented in [src/banksimfm/inference.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/inference.py).

## 11.1 `score_customer`

This function:

- accepts a customer history
- loads trained models if available
- returns a distress probability
- returns a distress label
- returns top risk drivers
- returns recent risk signals

If trained models are unavailable, it falls back to a heuristic probability using account-state risk factors.

## 11.2 `forecast_customer`

This function:

- accepts a customer history
- decodes future steps with the transformer when compatible trained artifacts exist
- keeps the account-state engine in the loop for financial consistency
- returns projected events
- returns projected balance path
- returns projected utilization path
- returns a forecast summary

Important note:

The current forecast is transformer-decoded in the main path and falls back to a heuristic path only if trained artifacts are missing or incompatible.

## 11.3 `simulate_intervention`

This function:

- scores the baseline scenario
- forecasts the baseline path
- applies an intervention policy to the account state
- reforecasts the adjusted scenario with the matching intervention token
- rescoring and reforecasting the adjusted scenario
- compares baseline and intervention outcomes

Supported interventions are:

- `reminder`
- `due_date_shift_7d`
- `temporary_overdraft_buffer`
- `installment_restructure`

This gives the project a practical what-if capability with learned intervention-conditioned generation inside the synthetic environment, while still being explicit that the result is not causal proof.

## 12. Deterministic Account-State Engine

The account-state logic is in [src/banksimfm/sim/engine.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/sim/engine.py).

This module is very important because it plays the role most similar to the MarS clearing mechanism.

## 12.1 What It Tracks

It updates:

- account balance
- utilized credit
- missed payments
- overdraft events
- failed debits
- due amount
- days to next due
- low-balance streak

## 12.2 Why It Matters

Without a state engine, generated events can become inconsistent or unrealistic. For example, a payment could appear without reducing balance, or a due amount could vanish without being paid.

The state engine ensures that:

- events have consequences
- the simulated path remains internally consistent
- scenario comparison makes financial sense

This is one of the strongest conceptual links between BankSimFM and MarS.

## 13. Streamlit Demo Application

The dashboard is implemented in [src/banksimfm/app/dashboard.py](/Users/abhishek/Desktop/Projects/regtech/src/banksimfm/app/dashboard.py) and launched from [app.py](/Users/abhishek/Desktop/Projects/regtech/app.py).

## 13.1 Overview Page

The Overview page currently shows:

- project summary
- KPI cards
- MarS mapping table
- distress rate by archetype
- customer distress distribution
- holdout metrics
- portfolio stress monitoring by segment and intervention
- representative customers

## 13.2 Customer Explorer

The Customer Explorer allows the user to:

- choose a customer
- review explicit balance and utilization charts
- review the event timeline
- inspect recent history
- view 30-day distress probability
- inspect top drivers and recent risk signals

## 13.3 What-If Simulator

The What-If Simulator allows the user to:

- choose a customer
- choose a forecast horizon
- choose an intervention
- compare baseline and intervention risk
- compare projected balance paths
- compare projected utilization paths
- inspect top forecasted negative events
- inspect scenario-level metrics
- inspect scenario notes and projected events

## 13.4 Model And Governance

The Model And Governance page shows:

- transformer-versus-LSTM architecture summary
- rationale for using the LSTM as the primary scorer and the transformer as the forecaster
- synthetic-data disclaimer
- privacy, fairness, explainability, reliability, and operational-risk notes
- fairness tables by income band, employment type, region, risk segment, and archetype
- simulation-quality and early-warning summaries from saved artifacts

## 14. Outcomes We Can Honestly Claim

Based on the current workspace, we can confidently say the project already demonstrates:

- a synthetic retail banking sequence dataset
- a MarS-inspired sequential modeling framework
- a transformer and LSTM benchmark pipeline
- 30-day distress scoring
- short-horizon event-path forecasting
- what-if intervention simulation
- a reproducible Streamlit analyst demo
- automated tests for core data and model pipeline behavior

## 15. What We Should Present Carefully

Some parts are implemented in a simplified way and should be described carefully:

- intervention impact is directional within a synthetic simulator rather than causal proof
- synthetic data is realistic enough for demo use, but not a substitute for real bank history
- current explainability is heuristic and timeline-based rather than a full model-interpretability framework

Being transparent about this will make the final report stronger, not weaker.

## 16. Why The Project Still Fits The Assignment Well

Even with these simplifications, the project fits the assignment very well because it demonstrates:

- technical understanding of the MarS design philosophy
- adaptation of a foundation-model idea to a new finance domain
- a workable sequential architecture and training flow
- multiple downstream use cases
- governance awareness
- a functioning prototype rather than only theory

That means the project is not just a concept note. It is a real prototype that supports the core story of the assignment.

## 17. Best Way To Explain The Project In One Paragraph

BankSimFM is a retail banking financial distress early-warning simulator inspired by the MarS financial market simulation framework. Instead of modeling order flow, it models a customer’s sequence of banking events such as salary credits, bill payments, card spend, missed dues, and overdraft-related activity. The project uses synthetic event histories, a transformer as the flagship sequence model, an LSTM as the baseline, a deterministic account-state engine for financially consistent simulation, and a Streamlit dashboard for analyst-facing scoring, forecasting, and intervention what-if analysis.

## 18. Best Way To Explain The Main Contribution

The main contribution of the project is showing that the MarS idea of sequence-based financial world modeling can be adapted from markets to retail banking by treating a customer’s financial life as an event sequence and using that sequence for distress detection, trajectory forecasting, and intervention testing.

## 19. What Is Still Missing If Someone Asks

If someone asks what would be the next improvement beyond the current prototype, the most accurate answer is:

1. run repeated-seed experiments and compare stability across training runs
2. improve the realism and calibration of synthetic intervention outcomes
3. expand economic-value measurement for collections, retention, and operations use cases
4. validate the dashboard and model behavior with a cleaner final report and presentation narrative
