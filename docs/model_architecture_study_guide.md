# BankSimFM Model Architecture Study Guide

This document is a **study guide**, not a submission document. Its purpose is to help you understand what we built, why we built it this way, and how to explain it clearly in your report, viva, or presentation.

If you remember only one sentence, remember this:

> BankSimFM treats a customer's financial life as a sequence of events, then uses a transformer-based model to score short-term distress risk, forecast likely future events, and simulate how interventions may change that path.

## 1. What Problem We Are Solving

The system is trying to answer three practical banking questions:

1. Is this customer likely to enter financial distress in the next 30 days?
2. What is likely to happen next in this customer's financial timeline?
3. If the bank takes an action, such as a due-date shift or restructuring, does the forecasted risk improve?

This is different from a traditional credit score or static risk model.

A static model usually looks at summary fields such as:

- average balance
- utilization
- income band
- number of missed payments

That is useful, but it can miss *how the customer got there*.

Our project assumes that distress is often a **sequence problem**:

- salary comes late
- rent and utilities hit first
- card spend continues
- credit-card due date arrives
- balance is already stressed
- payment fails or is missed

That sequence matters. So we built a **sequence model** instead of a purely tabular model.

## 2. The Big Idea In Simple Terms

You can think of BankSimFM as having five main layers:

1. **Synthetic world creation**
2. **Sequence data preparation**
3. **Model training**
4. **Forecasting and intervention simulation**
5. **Dashboard decision support**

At a very high level:

```text
Synthetic customers and event histories
        ->
convert them into training sequences
        ->
train transformer and LSTM models
        ->
use the transformer to score and forecast
        ->
show results in the dashboard
```

## 3. What Data We Built

Because this is a course project, we do not have real bank customer data. So we created a synthetic banking environment.

### 3.1 Customer Archetypes

The generator creates five broad customer types:

- `stable_salaried`
- `volatile_income`
- `high_obligation`
- `rising_utilization`
- `near_distress`

These are not exact templates. Each synthetic customer is slightly different within their archetype.

So two `near_distress` customers are not identical. They can differ in:

- income level
- volatility
- rent burden
- spending behavior
- repayment tendency
- recovery tendency
- starting liquidity

This matters because if the generator were too deterministic, the model would just memorize patterns instead of learning general sequence behavior.

### 3.2 Event Stream

Each customer produces a chronological event history with events such as:

- `salary_credit`
- `rent_payment`
- `utility_payment`
- `card_spend`
- `credit_card_payment_due`
- `credit_card_payment_made`
- `loan_emi_due`
- `loan_emi_paid`
- `loan_emi_missed`
- `failed_debit`
- `overdraft_event`
- `support_contact`

Each event also carries account-state information, for example:

- `balance_before`
- `balance_after`
- `credit_utilization`
- `days_to_next_due`
- intervention marker

So the model sees not just *what happened*, but also the surrounding financial state.

## 4. Why This Is Inspired By MarS

The MarS paper models markets as sequences of events and learns to simulate future trajectories.

We adapted that idea to retail banking:

- MarS models **market event sequences**
- BankSimFM models **customer financial-event sequences**

The analogy is:

- order flow in a market -> cash-flow / repayment behavior in a customer account
- market trajectory -> customer distress trajectory
- controllable market simulation -> intervention-conditioned customer simulation

So conceptually, we are building a **small customer-level financial world model**.

## 5. How The Data Becomes Model Input

The raw CSV-style event history is not directly fed into the model. It first goes through preprocessing.

### 5.1 Customer-Level Split

We split data by customer, not by event row.

That is important because otherwise the same customer could appear partly in train and partly in test, which would leak information and make evaluation unrealistically optimistic.

### 5.2 Fixed-Length History Windows

Each training sample is a fixed-length sequence of recent events, up to the configured context length.

You can think of a training example as:

- a window of past customer events
- plus dense state features
- plus targets for what happens next

### 5.3 What The Model Actually Sees

The model input includes:

- event-type tokens
- amount-related buckets
- balance-related state
- utilization
- days until next due payment
- intervention token
- internal due-state information

That last point is especially important.

We later discovered that interventions like `installment_restructure` do not make sense unless the model can “see” due-state changes properly. So the current version includes internal due-related features to preserve that signal.

## 6. Why We Built Two Models

We trained:

1. a **transformer**
2. an **LSTM baseline**

Before talking about their roles in BankSimFM, it helps to understand what each model is in simple terms.

### 6.1 What A Transformer Is

A transformer is a neural-network architecture for sequence modeling.

Its key idea is **attention**.

Attention means the model can look back across many earlier positions in the sequence and decide which past events matter most for understanding the current situation or predicting what comes next.

In simple language:

- it does not just remember one compressed summary of the past
- it can compare the current event with many earlier events
- it can decide that some earlier events matter more than others

For example, if the model sees:

- rent already deducted
- salary delayed or smaller than usual
- card spending still rising
- due payment approaching

it can learn that this combination is a strong distress pattern.

That is why transformers are powerful for long-range dependencies.

### 6.2 What A Causal Transformer Means

Our transformer is **causal**.

That means when it predicts the next step, it can only see the **past**, not the future.

So if the sequence is at event 50, the model can use events `1..50`, but not events `51+`.

This is important because:

- it matches real forecasting
- it avoids future leakage
- it supports autoregressive generation

Autoregressive generation means:

1. predict the next event
2. append it to the history
3. predict the next event after that
4. continue step by step

That is how our forecasting path works.

### 6.3 What An LSTM Is

LSTM stands for **Long Short-Term Memory**.

It is a type of recurrent neural network.

An LSTM reads the sequence one event at a time:

1. read an event
2. update internal memory
3. read the next event
4. update memory again
5. continue until the end

Its special gating mechanism decides:

- what to keep in memory
- what to forget
- what to pass forward

That makes it much stronger than a plain RNN for longer sequences.

### 6.4 Simple Difference Between Transformer And LSTM

A useful way to remember the difference is:

- **LSTM** reads from left to right and carries memory forward
- **Transformer** uses attention to look across many earlier events and decide what matters most

Another way to say it:

- LSTM is like reading a story page by page while remembering the important parts
- Transformer is like reading the story and being able to glance back at many earlier pages whenever needed

Both are sequence models, but transformers are usually more flexible for longer-range patterns and generation.

### 6.5 Transformer In This Project

The transformer is the main model now.

Its job is to:

- score current 30-day distress risk
- forecast likely future events
- generate intervention-conditioned future paths
- rescore simulated futures

Why use it?

Because transformers are strong at learning relationships across long sequences. A payment failure today may depend on interactions across many earlier events, not just the last one or two.

In BankSimFM specifically, the transformer consumes:

- event-type sequence tokens
- dense account-state features
- utilization and due-related signals
- intervention-aware information

And it produces multiple outputs:

- current distress probability
- next-event prediction
- next amount bucket
- next balance-delta bucket

So in this project, the transformer is doing two jobs at once:

1. it is a **risk model**
2. it is a **trajectory model**

That is why it fits the MarS-inspired idea so well.

### 6.6 LSTM In This Project

The LSTM is kept as a benchmark.

Its role is:

- provide a simpler sequence baseline
- help show whether the transformer is actually worth using

Earlier in the project, the LSTM was temporarily used as the primary scorer because the transformer was unstable. After tuning and fixing the intervention path, the transformer became stronger and is now the default live scorer.

That gives you a nice project story:

- we built both
- we compared them
- we promoted the better model based on actual results

In BankSimFM, the LSTM mainly answers:

- how strong is a simpler sequence baseline on the distress-classification task?

So the LSTM is still very important academically, even though it is no longer the main live scorer.

### 6.7 Why The Transformer Fits This Project Better

This is a very useful explanation to remember for viva or presentation questions.

The LSTM is good at sequence classification, but our project needs more than classification.

We need:

- live risk scoring
- future-event forecasting
- intervention-conditioned generation
- scenario rescoring

The transformer fits that full stack better because:

- it captures long-range event relationships more flexibly
- it naturally supports causal autoregressive decoding
- it can share one backbone across multiple related tasks

So the best summary is:

- **LSTM** = simpler sequence benchmark
- **Transformer** = stronger full-system model for scoring, forecasting, and simulation

## 7. What The Transformer Is Actually Learning

The transformer is not trained on just one task.

It is a **multitask model**.

It learns to predict:

- the **next event**
- the **next amount bucket**
- the **next balance-delta bucket**
- the **30-day distress label**

Why do this?

Because we want the model to understand both:

- the *behavioral sequence*
- the *risk outcome*

If we trained only for distress classification, the model might become a simple classifier without learning enough about trajectory dynamics.

If we trained only for next-event prediction, it might become a good generator but weak risk scorer.

So the multitask setup tries to teach both world understanding and risk detection.

## 8. What The LSTM Is Learning

The LSTM is trained mainly as a distress classifier.

It processes the same history windows but does not play the same generative role as the transformer.

So the LSTM is useful for answering:

- can a simpler sequential model classify distress reasonably well?

But it is not the best model for:

- controllable future decoding
- intervention-conditioned trajectory generation

That is one reason the transformer is a better fit for the MarS-inspired part of the project.

## 9. Why The Deterministic State Engine Exists

This is one of the most important pieces to understand.

The sequence model predicts events and risk, but we also need financial consistency.

For example:

- if the model forecasts a large debit, balance should go down
- if it forecasts a payment, utilization may change
- if an intervention shifts a due date, the internal due-state should change

A pure neural generator might create unrealistic or inconsistent paths.

So we use a **deterministic account-state engine** as the financial source of truth.

Its role is to update:

- balances
- utilization
- due timing
- repayment-related state

This makes the forecasted path more believable and more useful for simulation.

In MarS language, this engine plays a role similar to a clearing or state-update mechanism.

## 10. How Forecasting Works

Forecasting is now learned, not just rule-based.

The current flow is:

1. take observed customer history
2. convert it into model input
3. use the transformer to decode likely next events autoregressively
4. after each step, update financial state through the engine
5. continue until the forecast horizon is reached

So the forecast is a combination of:

- learned sequence decoding
- deterministic financial-state updates

That combination is one of the most important design choices in the project.

## 11. How Intervention Simulation Works

This is the part that makes the prototype feel more like a decision-support system than just a classifier.

We support four interventions:

- `reminder`
- `due_date_shift_7d`
- `temporary_overdraft_buffer`
- `installment_restructure`

### 11.1 Baseline Path

First, the model forecasts what happens if the bank does nothing.

### 11.2 Intervention Path

Then it:

- adjusts the internal account state
- sets the intervention token
- decodes a new future path under that intervention

### 11.3 Compare The Two

Finally, it compares:

- baseline risk
- intervention risk
- projected negative events
- balance/utilization trajectories

This is how the dashboard can tell an analyst:

- “this customer is risky”
- “this intervention appears better than the others”

## 12. A Very Important Fix We Made

At one stage, intervention usefulness looked broken because all the average risk reductions were zero.

The real problem was not that interventions were doing nothing.

The problem was that:

- intervention paths were different
- but the LSTM rescoring of simulated futures was too flat

So we fixed the design by using the **transformer as the scenario-risk scorer for simulated futures**.

That was an important conceptual correction:

- **observed current history** -> transformer live scorer
- **generated future scenario** -> transformer scenario rescoring

This made simulation metrics finally reflect the actual differences in generated paths.

## 13. Why The Transformer Is Now The Primary Live Scorer

The transformer is now the primary scorer because it currently performs better on the holdout set.

Current results:

- Transformer test AUC: `0.8342`
- Transformer test F1: `0.5828`
- Transformer precision: `0.6127`
- Transformer recall: `0.5556`

- LSTM test AUC: `0.8126`
- LSTM test F1: `0.5126`
- LSTM precision: `0.6458`
- LSTM recall: `0.4250`

How to explain this simply:

- the transformer gives the best overall balance of ranking power and classification quality
- the LSTM is still useful as a benchmark
- the project now uses the stronger model as the operational default

That is a clean and defensible story.

## 14. What The Metrics Mean In Plain English

### 14.1 AUC

AUC tells us how well the model ranks risky customers above non-risky customers.

Higher AUC means:

- if you randomly pick one distressed and one non-distressed customer,
- the model is more likely to assign a higher score to the distressed one

### 14.2 Precision

Precision answers:

- of the customers we flagged as risky, how many were actually risky?

### 14.3 Recall

Recall answers:

- of the actually risky customers, how many did we catch?

### 14.4 F1

F1 balances precision and recall.

This is often a good single summary metric when we care about both:

- not missing too many risky customers
- not flagging too many safe ones

## 15. How To Explain The Intervention Results

Current intervention summary:

- `due_date_shift_7d` helps on average
- `installment_restructure` helps most on average
- `temporary_overdraft_buffer` also helps on average
- `reminder` is negative on average

That does **not** mean the model is broken.

A good explanation is:

- not all interventions are equally strong
- lightweight reminder-only interventions may not help the most stressed customers
- structural interventions can produce bigger trajectory changes

This actually makes the system story more realistic.

If every intervention always helped, the simulation would look suspiciously simplistic.

## 16. What The Dashboard Is Really Showing

The dashboard is not just a UI layer. It reflects the system design.

### 16.1 Overview

Shows:

- portfolio-level distress patterns
- benchmark metrics
- cohort-level stress monitoring

### 16.2 Customer Explorer

Shows:

- current risk
- customer timeline
- balance behavior
- utilization behavior

### 16.3 What-If Simulator

Shows:

- baseline forecast
- intervention forecast
- risk delta
- negative-event summary

### 16.4 Collections Prioritization

Shows:

- which customers are most actionable
- current risk
- best intervention
- projected risk reduction

### 16.5 Model And Governance

Shows:

- architecture summary
- transformer vs LSTM comparison
- fairness notes
- privacy, reliability, and usage disclaimers

## 17. What Makes This A Good Course Project Technically

The project is strong because it is not just a static classifier demo.

It includes:

- synthetic world design
- sequence data engineering
- model comparison
- transformer-based forecasting
- intervention-conditioned generation
- simulation evaluation
- fairness analysis
- dashboard-based downstream applications

So it covers both:

- technical understanding of foundation-model ideas
- practical financial-services application design

## 18. What Its Limitations Are

You should also be ready to explain limits honestly.

### 18.1 Synthetic Data

The biggest limitation is that the environment is synthetic.

That means:

- patterns are plausible, but not truly validated against real banking behavior
- subgroup fairness results are illustrative, not production-grade

### 18.2 Not Causal

Intervention results are directional simulation outputs, not proof of real-world treatment effect.

### 18.3 Prototype Scale

This is a prototype, not a large production foundation model.

### 18.4 Stability Work Could Go Further

We have evaluation artifacts and repeated-run simulation checks, but a fuller multi-seed training study would strengthen the research story further.

## 19. How To Explain The Whole System In 30 Seconds

If you need a short viva answer, use something like this:

> We built a MarS-inspired retail banking sequence model that treats a customer's financial history as an ordered stream of events such as salary credits, bill payments, card spend, and repayment behavior. A causal transformer learns from these sequences to estimate 30-day distress risk, forecast likely future events, and simulate how bank interventions may change the trajectory. A deterministic account-state engine keeps the simulated paths financially coherent, and a Streamlit dashboard exposes the outputs for early warning, what-if analysis, portfolio monitoring, and collections prioritization.

## 20. How To Explain The Whole System In 2 Minutes

If you need a slightly longer answer:

> The core idea of BankSimFM is that financial distress is often a sequence problem rather than just a static risk-score problem. Instead of only looking at summary variables, we model the customer's financial life as a timeline of events: salary credits, recurring obligations, spending, repayment events, and balance deterioration. We generate synthetic customer-event histories, convert them into fixed-length training windows, and train both a transformer and an LSTM. The transformer is multitask: it predicts next events, amount and balance-related buckets, and 30-day distress. It is also used for learned forecasting and intervention-conditioned simulation. A deterministic account-state engine updates balances and utilization so the future paths remain financially realistic. We compare interventions like due-date shifts and restructuring in a what-if simulator, and we surface everything through a dashboard for analysts. The transformer now performs better than the LSTM on holdout metrics, so it has become the primary live scorer, while the LSTM remains the benchmark baseline.

## 21. Best Way To Study This

If you are revising for explanation rather than coding, focus on these five ideas:

1. **Why sequence modeling matters**
   - distress comes from event progression over time

2. **Why the transformer exists**
   - it models long-range dependencies and supports forecasting

3. **Why the LSTM exists**
   - it gives a simpler baseline for comparison

4. **Why the deterministic engine exists**
   - it keeps generated futures financially consistent

5. **Why intervention simulation matters**
   - it turns the system into a decision-support tool instead of just a classifier

If you can explain those five clearly, you already understand the heart of the project.
