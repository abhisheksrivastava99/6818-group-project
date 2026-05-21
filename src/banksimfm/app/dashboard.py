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


def _priority_tier(current_risk: float, risk_reduction: float) -> str:
    if current_risk >= 0.50 and risk_reduction >= 0.05:
        return "High"
    if current_risk >= 0.30 and risk_reduction > 0:
        return "Medium"
    return "Monitor"


def _build_collections_prioritization_frame(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    rows = []
    customer_meta = customers.set_index("customer_id").to_dict("index")
    grouped_events = events.sort_values(["customer_id", "event_timestamp"]).groupby("customer_id", sort=True)

    for customer_id, history in grouped_events:
        history = history.reset_index(drop=True)
        meta = customer_meta.get(customer_id, {})
        score = score_customer(history, horizon_days=horizon_days)
        best_intervention = None
        best_intervention_risk = float("inf")

        for intervention in INTERVENTIONS:
            result = simulate_intervention(history, intervention, horizon_days=horizon_days)
            intervention_risk = float(result.intervention.risk.distress_probability)
            if intervention_risk < best_intervention_risk:
                best_intervention_risk = intervention_risk
                best_intervention = intervention

        current_risk = float(score.distress_probability)
        risk_reduction = current_risk - best_intervention_risk
        top_driver = score.top_drivers[0] if score.top_drivers else "stable profile"
        rows.append(
            {
                "customer_id": customer_id,
                "current_risk": current_risk,
                "current_distress_label": "High Risk" if score.distress_label else "Lower Risk",
                "archetype": meta.get("archetype", "unknown"),
                "income_band": meta.get("income_band", "unknown"),
                "employment_type": meta.get("employment_type", "unknown"),
                "region": meta.get("region", "unknown"),
                "risk_segment": meta.get("risk_segment", "unknown"),
                "current_balance": float(history["balance_after"].iloc[-1]),
                "current_utilization": float(history["credit_utilization"].iloc[-1]),
                "top_driver": top_driver,
                "recommended_intervention": best_intervention,
                "predicted_risk_reduction": risk_reduction,
                "projected_post_intervention_risk": best_intervention_risk,
                "priority_tier": _priority_tier(current_risk, risk_reduction),
                "positive_risk_reduction": max(risk_reduction, 0.0),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["current_risk", "positive_risk_reduction", "projected_post_intervention_risk"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_collections_live_frame(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    rows = []
    customer_meta = customers.set_index("customer_id").to_dict("index")
    grouped_events = events.sort_values(["customer_id", "event_timestamp"]).groupby("customer_id", sort=True)

    for customer_id, history in grouped_events:
        history = history.reset_index(drop=True)
        meta = customer_meta.get(customer_id, {})
        score = score_customer(history, horizon_days=horizon_days)
        rows.append(
            {
                "customer_id": customer_id,
                "current_risk": float(score.distress_probability),
                "current_distress_label": "High Risk" if score.distress_label else "Lower Risk",
                "archetype": meta.get("archetype", "unknown"),
                "income_band": meta.get("income_band", "unknown"),
                "employment_type": meta.get("employment_type", "unknown"),
                "region": meta.get("region", "unknown"),
                "risk_segment": meta.get("risk_segment", "unknown"),
                "current_balance": float(history["balance_after"].iloc[-1]),
                "current_utilization": float(history["credit_utilization"].iloc[-1]),
                "top_driver": score.top_drivers[0] if score.top_drivers else "stable profile",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["current_risk", "current_balance"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner="Building collections prioritization view...")
def _cached_collections_prioritization_frame(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    return _build_collections_prioritization_frame(customers, events, horizon_days=horizon_days)


@st.cache_data(show_spinner="Scoring customer portfolio...")
def _cached_collections_live_frame(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    return _build_collections_live_frame(customers, events, horizon_days=horizon_days)


def _filter_collections_prioritization_frame(
    frame: pd.DataFrame,
    archetype: str,
    income_band: str,
    employment_type: str,
    region: str,
    risk_segment: str,
    min_current_risk: float,
) -> pd.DataFrame:
    filtered = frame.copy()
    filters = {
        "archetype": archetype,
        "income_band": income_band,
        "employment_type": employment_type,
        "region": region,
        "risk_segment": risk_segment,
    }
    for column, value in filters.items():
        if value != "All":
            filtered = filtered[filtered[column] == value]
    filtered = filtered[filtered["current_risk"] >= min_current_risk]
    return filtered.reset_index(drop=True)


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

    page = st.sidebar.radio(
        "Pages",
        ["Overview", "Customer Explorer", "What-If Simulator", "Collections Prioritization", "Model And Governance"],
    )

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

    elif page == "Collections Prioritization":
        st.subheader("Collections Prioritization")
        st.caption("Actionable outreach triage based on current transformer risk and best available intervention upside.")

        live_frame = _cached_collections_live_frame(bundle.customers.copy(), events.copy(), horizon_days=30)
        if live_frame.empty:
            st.info("Collections prioritization will appear once demo customers and artifacts are available.")
            return

        filter_cols = st.columns(6)
        archetype = filter_cols[0].selectbox("Archetype", ["All"] + sorted(bundle.customers["archetype"].dropna().unique().tolist()))
        income_band = filter_cols[1].selectbox("Income Band", ["All"] + sorted(bundle.customers["income_band"].dropna().unique().tolist()))
        employment_type = filter_cols[2].selectbox("Employment", ["All"] + sorted(bundle.customers["employment_type"].dropna().unique().tolist()))
        region = filter_cols[3].selectbox("Region", ["All"] + sorted(bundle.customers["region"].dropna().unique().tolist()))
        risk_segment = filter_cols[4].selectbox("Risk Segment", ["All"] + sorted(bundle.customers["risk_segment"].dropna().unique().tolist()))
        min_current_risk = filter_cols[5].slider("Min Current Risk", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

        filtered_live = _filter_collections_prioritization_frame(
            live_frame,
            archetype=archetype,
            income_band=income_band,
            employment_type=employment_type,
            region=region,
            risk_segment=risk_segment,
            min_current_risk=min_current_risk,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers Reviewed", f"{len(filtered_live)}")
        col2.metric(
            "High-Risk Candidates",
            f"{int((filtered_live['current_risk'] >= 0.50).sum())}",
        )
        col3.metric("Average Current Risk", f"{filtered_live['current_risk'].mean():.2%}" if not filtered_live.empty else "0.00%")
        col4.metric(
            "Avg Current Utilization",
            f"{filtered_live['current_utilization'].mean():.2%}" if not filtered_live.empty else "0.00%",
        )

        if filtered_live.empty:
            st.info("No customers match the current triage filters.")
        else:
            selected_customer_ids = filtered_live["customer_id"].tolist()
            selected_customers = bundle.customers[bundle.customers["customer_id"].isin(selected_customer_ids)].copy()
            selected_events = events[events["customer_id"].isin(selected_customer_ids)].copy()
            filtered = _cached_collections_prioritization_frame(selected_customers, selected_events, horizon_days=30)

            summary_col1, summary_col2 = st.columns(2)
            summary_col1.metric("High-Priority Customers", f"{int((filtered['priority_tier'] == 'High').sum())}")
            summary_col2.metric(
                "Average Best-Case Risk Reduction",
                f"{filtered['predicted_risk_reduction'].mean():.2%}" if not filtered.empty else "0.00%",
            )

            scatter = px.scatter(
                filtered,
                x="current_risk",
                y="predicted_risk_reduction",
                color="priority_tier",
                hover_data=["customer_id", "recommended_intervention", "top_driver"],
                title="Actionable Risk vs. Intervention Upside",
            )
            st.plotly_chart(scatter, use_container_width=True)

            st.caption("Customer IDs below can be copied into Customer Explorer or What-If Simulator for deeper review.")
            selected_customer = st.selectbox("Quick Customer Lookup", filtered["customer_id"].tolist())
            selected_row = filtered[filtered["customer_id"] == selected_customer].iloc[0]
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "customer_id": selected_row["customer_id"],
                            "priority_tier": selected_row["priority_tier"],
                            "current_risk": selected_row["current_risk"],
                            "recommended_intervention": selected_row["recommended_intervention"],
                            "predicted_risk_reduction": selected_row["predicted_risk_reduction"],
                            "projected_post_intervention_risk": selected_row["projected_post_intervention_risk"],
                            "top_driver": selected_row["top_driver"],
                        }
                    ]
                ),
                use_container_width=True,
            )

            display_columns = [
                "customer_id",
                "current_risk",
                "current_distress_label",
                "archetype",
                "risk_segment",
                "current_balance",
                "current_utilization",
                "top_driver",
                "recommended_intervention",
                "predicted_risk_reduction",
                "projected_post_intervention_risk",
                "priority_tier",
            ]
            st.markdown("### Ranked Outreach Queue")
            st.dataframe(filtered[display_columns], use_container_width=True)

    else:
        st.subheader("Model And Governance")
        st.markdown("### Architecture Summary")
        st.table(
            pd.DataFrame(
                [
                    ["Transformer", "Primary live distress scorer, learned forecaster, and intervention-conditioned scenario model"],
                    ["LSTM", "Baseline benchmark model retained for comparison in holdout metrics and fairness views"],
                ],
                columns=["Model", "Role"],
            )
        )

        st.markdown("### Model Rationale")
        st.write("The transformer is the MarS-inspired flagship sequence model and is now the default live distress scorer because the current saved holdout metrics favor it. The LSTM is retained as the baseline benchmark for comparison in metrics, fairness views, and governance review.")

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
        fairness_model = st.selectbox("Model", ["transformer", "lstm"])
        fairness_field = st.selectbox("Subgroup Field", ["income_band", "employment_type", "region", "risk_segment", "archetype"])
        fairness_frame = _flatten_fairness_metrics(fairness, fairness_model, fairness_field)
        if not fairness_frame.empty:
            st.dataframe(fairness_frame, use_container_width=True)
        else:
            st.info("Fairness metrics are generated after training.")
