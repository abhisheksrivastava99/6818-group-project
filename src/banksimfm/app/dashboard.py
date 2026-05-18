"""Streamlit dashboard for BankSimFM."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from banksimfm.eval.reporting import load_metrics_summary
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
    fig.update_traces(marker={"size": 9})
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


def run_dashboard() -> None:
    st.set_page_config(page_title="BankSimFM", layout="wide")
    st.title("BankSimFM")
    st.caption("Retail banking financial distress early-warning simulator inspired by MarS.")

    bundle = ensure_demo_artifacts()
    events = bundle.events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])
    representative = get_representative_customers(bundle)
    metrics = load_metrics_summary()

    page = st.sidebar.radio("Pages", ["Overview", "Customer Explorer", "What-If Simulator"])

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

        st.markdown("### Representative Customers")
        st.dataframe(representative, use_container_width=True)

    elif page == "Customer Explorer":
        st.subheader("Customer Explorer")
        customer_id = st.selectbox("Customer", representative["customer_id"].tolist())
        history = events[events["customer_id"] == customer_id].sort_values("event_timestamp")
        score = score_customer(history, horizon_days=30)

        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(_timeline_chart(history), use_container_width=True)
            st.dataframe(history.tail(25), use_container_width=True)
        with right:
            st.metric("30-Day Distress Probability", f"{score.distress_probability:.2%}")
            st.metric("Distress Label", "High Risk" if score.distress_label else "Lower Risk")
            st.markdown("### Top Drivers")
            for driver in score.top_drivers:
                st.write(f"- {driver}")
            st.markdown("### Recent Signals")
            for signal in score.recent_risk_signals:
                st.write(f"- {signal}")

    else:
        st.subheader("What-If Simulator")
        customer_id = st.selectbox("Customer", representative["customer_id"].tolist(), key="sim_customer")
        history = events[events["customer_id"] == customer_id].sort_values("event_timestamp")
        horizon = st.select_slider("Horizon", options=[30, 60, 90], value=30)
        intervention = st.selectbox("Intervention", INTERVENTIONS)
        result = simulate_intervention(history, intervention, horizon_days=horizon)

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Risk", f"{result.baseline.risk.distress_probability:.2%}")
        col2.metric("Intervention Risk", f"{result.intervention.risk.distress_probability:.2%}")
        col3.metric("Risk Delta", f"{result.risk_delta:.2%}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _balance_path_chart(result.baseline.forecast.balance_path, result.intervention.forecast.balance_path),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                _util_path_chart(result.baseline.forecast.utilization_path, result.intervention.forecast.utilization_path),
                use_container_width=True,
            )

        st.markdown("### Scenario Notes")
        for note in result.scenario_differences:
            st.write(f"- {note}")

        st.markdown("### Forecast Samples")
        forecast_frame = pd.DataFrame(result.intervention.forecast.projected_events)
        st.dataframe(forecast_frame, use_container_width=True)
