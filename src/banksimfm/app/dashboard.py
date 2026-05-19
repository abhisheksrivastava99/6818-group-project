"""Streamlit dashboard for BankSimFM."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from banksimfm.eval.reporting import load_fairness_summary, load_metrics_summary, load_simulation_summary
from banksimfm.inference import INTERVENTIONS, ensure_demo_artifacts, forecast_customer, get_representative_customers, score_customer, simulate_intervention


def _timeline_chart(history: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        history,
        x="event_timestamp",
        y="balance_after",
        color="event_type",
        hover_data=["amount", "credit_utilization", "distress_label_30d"],
        title="Customer Event Timeline",
    )
    fig.update_traces(marker={"size": 8})
    return fig


def _balance_history_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["event_timestamp"], y=history["balance_after"], mode="lines+markers", name="Balance"))
    fig.update_layout(title="Balance Over Time", xaxis_title="Time", yaxis_title="Balance")
    return fig


def _util_history_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["event_timestamp"], y=history["credit_utilization"], mode="lines+markers", name="Utilization"))
    fig.update_layout(title="Credit Utilization Over Time", xaxis_title="Time", yaxis_title="Utilization")
    return fig


def _balance_path_chart(baseline: list[float], intervention: list[float] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=baseline, mode="lines+markers", name="Baseline"))
    if intervention is not None:
        fig.add_trace(go.Scatter(y=intervention, mode="lines+markers", name="Intervention"))
    fig.update_layout(title="Projected Balance Path", xaxis_title="Forecast Step", yaxis_title="Balance")
    return fig


def _util_path_chart(baseline: list[float], intervention: list[float] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=baseline, mode="lines+markers", name="Baseline"))
    if intervention is not None:
        fig.add_trace(go.Scatter(y=intervention, mode="lines+markers", name="Intervention"))
    fig.update_layout(title="Projected Utilization Path", xaxis_title="Forecast Step", yaxis_title="Utilization")
    return fig


def _flatten_fairness_metrics(fairness_metrics: dict, model_name: str, field: str) -> pd.DataFrame:
    rows = []
    model_metrics = fairness_metrics.get(model_name, {}).get(field, {})
    for segment_value, metrics in model_metrics.items():
        rows.append({"segment_value": segment_value, **metrics})
    return pd.DataFrame(rows)


def _top_negative_events_frame(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(summary.get("top_negative_events", []))


def run_dashboard() -> None:
    st.set_page_config(page_title="BankSimFM", layout="wide")
    st.title("BankSimFM")
    st.caption("Retail banking financial distress early-warning simulator inspired by MarS.")
    st.info("Synthetic demo data only. This prototype is for coursework and scenario analysis, not production credit decisioning.")

    bundle = ensure_demo_artifacts()
    events = bundle.events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])
    representative = get_representative_customers(bundle)
    metrics = load_metrics_summary()
    fairness = load_fairness_summary()
    simulation = load_simulation_summary()

    page = st.sidebar.radio("Pages", ["Overview", "Customer Explorer", "What-If Simulator", "Model And Governance"])

    if page == "Overview":
        st.subheader("Project Summary")
        col1, col2, col3, col4 = st.columns(4)
        customer_distress = bundle.customers["customer_distress_label"].fillna(0)
        col1.metric("Customers", f"{bundle.customers['customer_id'].nunique()}")
        col2.metric("Events", f"{len(events):,}")
        col3.metric("Distressed Customers", f"{int(customer_distress.sum())}")
        col4.metric("Avg Events / Customer", f"{events.groupby('customer_id').size().mean():.1f}")

        st.markdown("### MarS Mapping")
        st.table(
            pd.DataFrame(
                [
                    ["Order sequence", "Customer financial-event sequence"],
                    ["Market trajectory", "Customer cash-flow and distress trajectory"],
                    ["Controllable generation", "Intervention-conditioned simulation"],
                ],
                columns=["MarS Concept", "BankSimFM Analogue"],
            )
        )

        left, right = st.columns(2)
        with left:
            cohort = bundle.customers.groupby("archetype")["customer_distress_label"].mean().reset_index()
            st.plotly_chart(px.bar(cohort, x="archetype", y="customer_distress_label", title="Distress Rate by Archetype"), use_container_width=True)
        with right:
            distress_hist = events.groupby("customer_id")["distress_label_30d"].max().reset_index()
            st.plotly_chart(px.histogram(distress_hist, x="distress_label_30d", nbins=2, title="Customer Distress Distribution"), use_container_width=True)

        st.markdown("### Holdout Metrics")
        if metrics:
            metrics_frame = pd.DataFrame(
                [
                    {"model": model_name, "split": split_name, **split_metrics}
                    for model_name, split_metrics_map in metrics.items()
                    for split_name, split_metrics in split_metrics_map.items()
                ]
            )
            st.dataframe(metrics_frame, use_container_width=True)
        else:
            st.info("Metrics are generated after the first training run.")

        st.markdown("### Portfolio Stress Monitoring")
        segment_options = ["archetype", "income_band", "employment_type", "region", "risk_segment"]
        selected_dimension = st.selectbox("Segment Dimension", segment_options)
        selected_intervention = st.selectbox("Intervention Scenario", INTERVENTIONS, key="portfolio_intervention")
        portfolio_rows = simulation.get("intervention_usefulness", {}).get("portfolio_by_segment", {}).get(selected_dimension, [])
        portfolio_frame = pd.DataFrame(portfolio_rows)
        if not portfolio_frame.empty:
            portfolio_frame = portfolio_frame[portfolio_frame["intervention"] == selected_intervention]
            st.plotly_chart(
                px.bar(
                    portfolio_frame,
                    x="segment_value",
                    y="average_risk_reduction",
                    color="average_intervention_risk",
                    title=f"Average Risk Reduction by {selected_dimension}",
                ),
                use_container_width=True,
            )
            st.dataframe(portfolio_frame, use_container_width=True)
        else:
            st.info("Portfolio stress metrics will appear after training artifacts are generated.")

        st.markdown("### Representative Customers")
        st.dataframe(representative, use_container_width=True)

    elif page == "Customer Explorer":
        st.subheader("Customer Explorer")
        customer_id = st.selectbox("Customer", sorted(bundle.customers["customer_id"].tolist()))
        customer_meta = bundle.customers[bundle.customers["customer_id"] == customer_id].iloc[0]
        history = events[events["customer_id"] == customer_id].sort_values("event_timestamp")
        score = score_customer(history, horizon_days=30)

        st.caption(
            f"Archetype: {customer_meta['archetype']} | Income Band: {customer_meta['income_band']} | "
            f"Employment: {customer_meta['employment_type']} | Region: {customer_meta['region']} | "
            f"Risk Segment: {customer_meta['risk_segment']}"
        )

        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(_balance_history_chart(history), use_container_width=True)
            st.plotly_chart(_util_history_chart(history), use_container_width=True)
            st.plotly_chart(_timeline_chart(history), use_container_width=True)
            st.dataframe(history.tail(25), use_container_width=True)
        with right:
            st.metric("30-Day Distress Probability", f"{score.distress_probability:.2%}")
            st.metric("Distress Label", "High Risk" if score.distress_label else "Lower Risk")
            st.metric("Current Balance", f"{history['balance_after'].iloc[-1]:.2f}")
            st.metric("Current Utilization", f"{history['credit_utilization'].iloc[-1]:.2%}")
            st.markdown("### Top Drivers")
            for driver in score.top_drivers:
                st.write(f"- {driver}")
            st.markdown("### Recent Signals")
            for signal in score.recent_risk_signals:
                st.write(f"- {signal}")

    elif page == "What-If Simulator":
        st.subheader("What-If Simulator")
        customer_id = st.selectbox("Customer", sorted(bundle.customers["customer_id"].tolist()), key="sim_customer")
        history = events[events["customer_id"] == customer_id].sort_values("event_timestamp")
        horizon = st.select_slider("Horizon", options=[30, 60, 90], value=30)
        intervention = st.selectbox("Intervention", INTERVENTIONS)
        result = simulate_intervention(history, intervention, horizon_days=horizon)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline Risk", f"{result.baseline.risk.distress_probability:.2%}")
        col2.metric("Intervention Risk", f"{result.intervention.risk.distress_probability:.2%}")
        col3.metric("Risk Delta", f"{result.risk_delta:.2%}")
        col4.metric(
            "Negative Events Delta",
            f"{result.intervention.forecast.forecast_summary['projected_negative_event_count'] - result.baseline.forecast.forecast_summary['projected_negative_event_count']:+d}",
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(_balance_path_chart(result.baseline.forecast.balance_path, result.intervention.forecast.balance_path), use_container_width=True)
            st.plotly_chart(_util_path_chart(result.baseline.forecast.utilization_path, result.intervention.forecast.utilization_path), use_container_width=True)
        with right:
            st.markdown("### Baseline Negative Events")
            st.dataframe(_top_negative_events_frame(result.baseline.forecast.forecast_summary), use_container_width=True)
            st.markdown("### Intervention Negative Events")
            st.dataframe(_top_negative_events_frame(result.intervention.forecast.forecast_summary), use_container_width=True)

        st.markdown("### Scenario Notes")
        for note in result.scenario_differences:
            st.write(f"- {note}")

        scenario_metrics = pd.DataFrame(
            [
                {"scenario": "baseline", **result.baseline.forecast.forecast_summary.get("scenario_metrics", {})},
                {"scenario": "intervention", **result.intervention.forecast.forecast_summary.get("scenario_metrics", {})},
            ]
        )
        st.markdown("### Scenario Metrics")
        st.dataframe(scenario_metrics, use_container_width=True)

        st.markdown("### Forecast Samples")
        forecast_frame = pd.DataFrame(result.intervention.forecast.projected_events)
        st.dataframe(forecast_frame, use_container_width=True)

    else:
        st.subheader("Model And Governance")
        st.markdown("### Architecture Summary")
        st.table(
            pd.DataFrame(
                [
                    ["Transformer", "Learns next-event dynamics, amount buckets, balance-delta buckets, and intervention-conditioned forecasting"],
                    ["LSTM", "Primary 30-day distress scorer in the current demo because it performs best on holdout classification"],
                ],
                columns=["Model", "Role"],
            )
        )

        st.markdown("### Model Rationale")
        st.write("The transformer is the MarS-inspired sequence model used for learned forecasting and scenario generation. The LSTM is retained as the operational primary distress classifier because it currently gives the strongest balanced holdout classification metrics on the tuned synthetic dataset.")

        st.markdown("### Governance Notes")
        st.write("- Privacy: synthetic customer-event data only.")
        st.write("- Fairness: subgroup metrics are tracked across income band, employment type, region, risk segment, and archetype.")
        st.write("- Explainability: the app surfaces timeline-based risk drivers and scenario deltas instead of raw scores alone.")
        st.write("- Reliability: holdout metrics, repeated-run stability, and scenario realism summaries are generated in artifacts.")
        st.write("- Operational risk: false positives can create unnecessary outreach, while false negatives can miss customers who need support.")
        st.write("- Limitation: intervention effects are directional and not causal proof.")

        st.markdown("### Early-Warning And Simulation Summary")
        early_warning = simulation.get("early_warning", {})
        simulation_quality = simulation.get("simulation_quality", {})
        stability = simulation.get("stability", {})
        if early_warning:
            st.json({"early_warning": early_warning, "simulation_quality": simulation_quality, "stability": stability})
        else:
            st.info("Simulation evaluation artifacts are generated after training.")

        st.markdown("### Fairness Metrics")
        fairness_model = st.selectbox("Model", ["lstm", "transformer"])
        fairness_field = st.selectbox("Subgroup Field", ["income_band", "employment_type", "region", "risk_segment", "archetype"])
        fairness_frame = _flatten_fairness_metrics(fairness, fairness_model, fairness_field)
        if not fairness_frame.empty:
            st.dataframe(fairness_frame, use_container_width=True)
        else:
            st.info("Fairness metrics are generated after training.")
