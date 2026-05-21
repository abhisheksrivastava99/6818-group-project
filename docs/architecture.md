# BankSimFM Architecture And Diagram Source

This document stores the Mermaid source for the main BankSimFM figures used in the report and slides. Each section includes a caption, a short explanation, the Mermaid diagram, and the intended report reference.

## Figure 1. MarS-To-BankSimFM Concept Map

**Caption:** Conceptual mapping from the MarS financial market simulation framework to the BankSimFM retail banking prototype.

This figure explains how the project inherits MarS design ideas while changing the modeled world from markets to customer financial-event sequences. It is intended to help readers understand the innovation thesis quickly before reading implementation details.

**Referenced in report:** Section 2, *Innovation Thesis And MarS Alignment*

```mermaid
flowchart LR
    subgraph M["MarS"]
        M1["Order sequence"]
        M2["Market trajectory"]
        M3["Generative sequence model"]
        M4["Clearing mechanism"]
        M5["Controllable simulation"]
        M6["Market-risk detection"]
    end

    subgraph B["BankSimFM"]
        B1["Customer financial-event sequence"]
        B2["Cash-flow and distress trajectory"]
        B3["Causal transformer over banking events"]
        B4["Deterministic account-state engine"]
        B5["Intervention-conditioned simulation"]
        B6["30-day distress warning"]
    end

    M1 --> B1
    M2 --> B2
    M3 --> B3
    M4 --> B4
    M5 --> B5
    M6 --> B6
```

## Figure 2. BankSimFM System Architecture

**Caption:** End-to-end system architecture from synthetic data generation to dashboard consumption.

This figure shows the main implementation blocks that now exist in the repository, including the transformer as primary live scorer, the LSTM baseline, the deterministic engine, and the dashboard. It is the main high-level architecture figure for the report.

**Referenced in report:** Section 4, *Model Architecture And Training Approach*

```mermaid
flowchart TD
    A["Synthetic customer profiles and archetypes"]
    B["Synthetic event generator"]
    C["Chronological event table and customer metadata"]
    D["Preprocessing, bucketing, and customer-level split"]
    E["History windows and intervention-augmented windows"]
    F["Transformer: primary live scorer, forecaster, scenario model"]
    G["LSTM: benchmark baseline classifier"]
    H["Deterministic account-state engine"]
    I["Saved artifacts: checkpoints, metrics, fairness, simulation"]
    J["Inference APIs: score, forecast, simulate"]
    K["Streamlit dashboard"]

    A --> B --> C --> D --> E
    E --> F
    E --> G
    F --> I
    G --> I
    F --> J
    H --> J
    I --> K
    J --> K
```

## Figure 3. Training And Evaluation Flow

**Caption:** Training and evaluation flow for the transformer and LSTM models.

This figure emphasizes how raw synthetic events become model-ready windows, how the two models are trained, and how the evaluation artifacts are produced. It is useful for the methodology section and for presentation slides.

**Referenced in report:** Section 4, *Model Architecture And Training Approach*

```mermaid
flowchart LR
    A["Synthetic customers and events"]
    B["Chronological sort and feature engineering"]
    C["Customer-level train / validation / test split"]
    D["Fixed-length history windows"]
    E["Intervention-augmented continuation windows"]
    F["Transformer multitask training"]
    G["LSTM distress-classifier training"]
    H["Validation thresholds and checkpoint selection"]
    I["metrics.json"]
    J["fairness_metrics.json"]
    K["simulation_metrics.json"]

    A --> B --> C --> D
    C --> E
    D --> F
    E --> F
    D --> G
    F --> H
    G --> H
    H --> I
    H --> J
    H --> K
```

## Figure 4. Baseline And Intervention Simulation Flow

**Caption:** Baseline and intervention simulation flow for what-if analysis.

This figure shows how observed history is scored, how baseline and intervention paths are decoded, how the deterministic state engine preserves financial consistency, and how the transformer rescoring logic supports scenario comparison.

**Referenced in report:** Section 5, *Forecasting, Scoring, And Intervention Simulation*

```mermaid
flowchart TD
    A["Observed customer history"]
    B["Transformer live scoring"]
    C["Baseline forecast decode"]
    D["Intervention-adjusted account state"]
    E["Intervention-conditioned forecast decode"]
    F["Deterministic state updates"]
    G["Transformer scenario rescoring"]
    H["Baseline vs intervention comparison"]

    A --> B
    A --> C
    A --> D
    C --> F
    D --> E
    E --> F
    F --> G
    B --> H
    G --> H
```

## Figure 5. Analyst Workflow And Collections Prioritization

**Caption:** Analyst-facing workflow across scoring, simulation, and collections prioritization.

This figure translates the technical system into a practical workflow for a risk or collections analyst. It highlights how the dashboard can move from monitoring to customer review to intervention comparison and then to ranked outreach action.

**Referenced in report:** Section 7, *Downstream Applications And Business Value*

```mermaid
flowchart LR
    A["Overview page: portfolio monitoring"]
    B["Customer Explorer: review timeline, balance, utilization"]
    C["Live distress score"]
    D["What-If Simulator: compare interventions"]
    E["Collections Prioritization: ranked outreach queue"]
    F["Human review and operational action"]

    A --> B --> C --> D --> E --> F
```

## Figure 6. Governance And Monitoring Control Loop

**Caption:** Governance and monitoring control loop for a responsible deployment path.

This figure summarizes the report's governance position. The prototype can inform analyst decisions, but it requires fairness review, monitoring, human oversight, and periodic refresh in any real deployment scenario.

**Referenced in report:** Section 8, *Governance, Risks, And Controls*

```mermaid
flowchart LR
    A["Model training and refresh"]
    B["Holdout evaluation and threshold review"]
    C["Fairness and reliability review"]
    D["Dashboard use by analysts"]
    E["Human-in-the-loop intervention decisions"]
    F["Operational monitoring and feedback"]

    A --> B --> C --> D --> E --> F --> A
```

## Figure 7. Dashboard Page Map

**Caption:** BankSimFM Streamlit dashboard page structure.

This optional figure helps presentation audiences see how the implemented prototype is organized. It is useful when giving a demo or summarizing the product surface quickly.

**Referenced in report:** Optional appendix or presentation materials

```mermaid
flowchart TD
    A["BankSimFM Dashboard"]
    A --> B["Overview"]
    A --> C["Customer Explorer"]
    A --> D["What-If Simulator"]
    A --> E["Collections Prioritization"]
    A --> F["Model And Governance"]
```

## Figure 8. Public API And Artifact Relationships

**Caption:** Relationship between the public APIs and the main saved artifacts.

This optional figure helps connect the implementation interfaces to the saved model and metric outputs. It is useful for readers who want a compact view of how runtime inference and offline evaluation relate to each other.

**Referenced in report:** Optional appendix or technical notes

```mermaid
flowchart LR
    A["score_customer(history, horizon_days=30)"]
    B["forecast_customer(history, horizon_days=30)"]
    C["simulate_intervention(history, intervention_type, horizon_days=30)"]
    D["transformer.pt"]
    E["lstm.pt"]
    F["metrics.json"]
    G["fairness_metrics.json"]
    H["simulation_metrics.json"]

    D --> A
    D --> B
    D --> C
    E --> A
    F --> A
    F --> C
    G --> A
    H --> C
```
