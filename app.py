"""
CloudWise — Google Cloud CUD & SUD Optimization Advisor
Streamlit Application for Google Cloud Associate Cloud Engineer Hackathon.

A modern FinOps advisor analyzing Compute Engine usage to evaluate:
- Sustained Use Discounts (SUD)
- Committed Use Discounts (CUD)
- Risk simulations (60% to 120% stress testing)
- Transparent confidence scoring & when to WAIT for more data
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure src/ is importable
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from src.data_loader import load_usage_data, get_sample_csv_path
from src.sud_analysis import analyze_sud, GCP_SUD_FAMILY_POLICY
from src.cud_analysis import analyze_cud_portfolio, DEFAULT_CUD_RATES
from src.risk_engine import (
    generate_portfolio_recommendations,
    simulate_commitment_stress,
    STRESS_MULTIPLIERS,
)
from src.report_generator import (
    build_recommendations_dataframe,
    generate_executive_summary_kpis,
    export_recommendations_to_csv,
)

# ---------------------------------------------------------
# Page Configuration & Custom FinOps Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="CloudWise — GCP CUD & SUD Advisor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A73E8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #5F6368;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #E8F0FE;
        border-left: 5px solid #1A73E8;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #174EA6;
    }
    .warning-banner {
        background-color: #FEF7E0;
        border-left: 5px solid #F9AB00;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #B06000;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(60,64,67, 0.08);
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #202124;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #5F6368;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-commit {
        background-color: #E6F4EA;
        color: #137333;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .badge-flexible {
        background-color: #E8F0FE;
        color: #1A73E8;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .badge-wait {
        background-color: #FEF7E0;
        color: #B06000;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar: Data Source & FinOps Parameters
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://www.gstatic.com/images/branding/product/2x/google_cloud_64dp.png", width=50)
    st.title("CloudWise Advisor")
    st.caption("Google Cloud FinOps MVP | Associate Cloud Engineer Hackathon")
    st.divider()

    st.subheader("1. Data Ingestion")
    data_source_option = st.radio(
        "Select Usage Data Source:",
        ["Bundled Synthetic Dataset (90 Days)", "Upload Custom Usage CSV"],
        index=0,
    )

    uploaded_file = None
    if data_source_option == "Upload Custom Usage CSV":
        uploaded_file = st.file_uploader(
            "Upload Compute Usage CSV",
            type=["csv"],
            help="Schema: date, project_id, resource_id, machine_type, region, usage_hours, on_demand_cost, existing_commitment, sud_eligible, history_days",
        )

    st.divider()
    st.subheader("2. Illustrative Pricing Rates")
    st.caption("Rates are illustrative and configurable for scenario modeling.")

    rate_1yr = st.slider("1-Year CUD Discount Rate", min_value=0.00, max_value=0.80, value=0.37, step=0.01, format="%d%%")
    rate_3yr = st.slider("3-Year CUD Discount Rate", min_value=0.00, max_value=0.80, value=0.55, step=0.01, format="%d%%")
    conserv_cov = st.slider("Conservative Target Coverage", min_value=0.00, max_value=1.00, value=0.70, step=0.05, format="%d%%")

    enforce_policy = st.checkbox(
        "Enforce GCP Machine Family SUD Policy",
        value=True,
        help="Checks machine types (e.g. E2, C3 are ineligible for SUD in GCP, overriding synthetic flags).",
    )

    custom_rates = {
        "cud_1yr": float(rate_1yr),
        "cud_3yr": float(rate_3yr),
        "conservative_coverage": float(conserv_cov),
    }

    st.divider()
    st.markdown("### 📘 Model Disclaimers")
    st.info(
        "**Synthetic Data**: Bundled records are synthetic and for demonstration.\n\n"
        "**Estimates Only**: Outputs are analytical recommendations, not official Google Cloud billing commitments or binding contracts."
    )


# ---------------------------------------------------------
# Data Ingestion & Validation
# ---------------------------------------------------------
if data_source_option == "Bundled Synthetic Dataset (90 Days)":
    sample_path = get_sample_csv_path()
    if not sample_path.exists():
        st.error(f"Sample dataset file not found at expected path: {sample_path}")
        st.stop()
    data_res = load_usage_data(sample_path)
    dataset_mode_label = "Bundled Deterministic Synthetic Dataset (468 records, 6 representative workloads)"
else:
    if uploaded_file is None:
        st.warning("Please upload a usage CSV file or switch to the bundled synthetic dataset.")
        st.stop()
    data_res = load_usage_data(uploaded_file)
    dataset_mode_label = f"Uploaded Custom Dataset ({uploaded_file.name})"

if not data_res.is_valid:
    st.error("❌ Data Validation Failed!")
    for err in data_res.errors:
        st.error(f"• {err}")
    if data_res.warnings:
        with st.expander("Validation Warnings"):
            for w in data_res.warnings:
                st.warning(f"• {w}")
    st.stop()

df = data_res.df

# Render any non-critical data loader warnings
if data_res.warnings:
    with st.expander(f"⚠️ Data Loader Audit Notices ({len(data_res.warnings)})", expanded=False):
        for w in data_res.warnings:
            st.caption(f"• {w}")


# ---------------------------------------------------------
# Execute Analysis Modules
# ---------------------------------------------------------
sud_report = analyze_sud(df, enforce_gcp_family_policy=enforce_policy)
cud_report = analyze_cud_portfolio(df, rates=custom_rates, enforce_gcp_family_policy=enforce_policy)
recommendations = generate_portfolio_recommendations(df, rates=custom_rates, enforce_gcp_family_policy=enforce_policy)
exec_kpis = generate_executive_summary_kpis(df, recommendations, cud_report)
recs_df = build_recommendations_dataframe(recommendations)


# ---------------------------------------------------------
# Top Header & Disclaimers
# ---------------------------------------------------------
st.markdown('<div class="main-title">CloudWise — Google Cloud CUD & SUD Optimization Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Algorithmic FinOps intelligence to evaluate Sustained Use Discounts, Committed Use Discounts, '
    'underutilization exposure, and evidence-based wait recommendations.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="disclaimer-banner">'
    f'<strong>Notice:</strong> Operating in <em>{dataset_mode_label}</em>. '
    f'All pricing figures and discount curves are modeled estimates for hackathon evaluation and do not represent '
    f'official Google Cloud billing quotes or binding contractual commitments.'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_overview, tab_analytics, tab_discounts, tab_risk, tab_recs, tab_methodology = st.tabs([
    "📊 Executive Overview",
    "📈 Usage Analytics",
    "🏷️ SUD & CUD Analysis",
    "⚡ Risk Simulator (60%–120%)",
    "📋 Recommendations & Export",
    "📘 Methodology & Audit Log",
])


# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tab_overview:
    st.subheader("Portfolio FinOps Summary")

    # Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Modeled Baseline Cost</div>'
            f'<div class="metric-value">${exec_kpis["total_on_demand_modeled"]:,.2f}</div>'
            f'<span style="color:#5F6368;font-size:0.8rem;">{data_res.summary["date_range"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        obs_sud = sud_report.summary_metrics["total_observed_sud_savings"]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Observed SUD Credits</div>'
            f'<div class="metric-value" style="color:#137333;">${obs_sud:,.2f}</div>'
            f'<span style="color:#137333;font-size:0.8rem;">+{sud_report.summary_metrics["portfolio_effective_sud_discount_pct"]:.1f}% automatic savings</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        inc_monthly = exec_kpis["total_monthly_incremental_savings"]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Est. Incremental Net Savings</div>'
            f'<div class="metric-value" style="color:#1A73E8;">${inc_monthly:,.2f}<span style="font-size:1rem;color:#5F6368;">/mo</span></div>'
            f'<span style="color:#1A73E8;font-size:0.8rem;">${exec_kpis["total_annual_incremental_savings"]:,.2f}/yr potential net ROI</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col4:
        total_w = exec_kpis["total_workloads_analyzed"]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Workloads Evaluated</div>'
            f'<div class="metric-value">{total_w}</div>'
            f'<span style="color:#5F6368;font-size:0.8rem;">{exec_kpis["actions_breakdown"].get("COMMIT", 0)} Commit | {exec_kpis["actions_breakdown"].get("WAIT", 0)} Wait | {exec_kpis["actions_breakdown"].get("FLEXIBLE", 0)} Flexible</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Strategy & Decision Distribution Charts
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### Recommended Actions Breakdown")
        action_counts = pd.DataFrame(
            [{"Action Category": k, "Count": v} for k, v in exec_kpis["actions_breakdown"].items()]
        )
        cat_color_map = {
            "COMMIT": "#34A853",
            "FLEXIBLE": "#4285F4",
            "WAIT": "#FBBC04",
            "NO_CHANGE": "#9AA0A6",
        }
        fig_actions = px.pie(
            action_counts,
            names="Action Category",
            values="Count",
            color="Action Category",
            color_discrete_map=cat_color_map,
            hole=0.45,
        )
        fig_actions.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_actions, use_container_width=True)

    with c_right:
        st.markdown("#### Confidence Level Distribution")
        conf_counts = pd.DataFrame(
            [{"Confidence": k, "Count": v} for k, v in exec_kpis["confidence_breakdown"].items()]
        )
        conf_color_map = {
            "High": "#34A853",
            "Medium": "#4285F4",
            "Low": "#EA4335",
            "Insufficient Evidence": "#FBBC04",
        }
        fig_conf = px.bar(
            conf_counts,
            x="Confidence",
            y="Count",
            color="Confidence",
            color_discrete_map=conf_color_map,
            text="Count",
        )
        fig_conf.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), yaxis_title="Number of Workloads")
        st.plotly_chart(fig_conf, use_container_width=True)

    # Executive Guidance Callout
    st.markdown("#### 💡 Key FinOps Guidance")
    has_wait = exec_kpis["actions_breakdown"].get("WAIT", 0)
    if has_wait > 0:
        st.warning(
            f"⚠️ **Caution:** {has_wait} workload(s) have insufficient history (<30 days). "
            "CloudWise actively advises **WAITING** to accumulate additional telemetry rather than making premature multi-year financial commitments."
        )
    if exec_kpis["total_monthly_incremental_savings"] <= 0.0:
        st.info(
            "ℹ️ **$0.00 Incremental Net Savings:** At current slider settings (0% discount, 0% target coverage, or when CUD discounts "
            "do not surpass existing automatic SUD credits), commitments provide no net pricing advantage over current operations. "
            "CloudWise advises maintaining 100% operational agility on On-Demand / SUD."
        )


# =========================================================
# TAB 2: USAGE ANALYTICS
# =========================================================
with tab_analytics:
    st.subheader("Compute Usage & Volatility Analytics")

    # Workload Filter
    all_workloads = sorted(df["resource_id"].unique().tolist())
    selected_workload_filter = st.selectbox(
        "Filter by Workload (or view all):",
        ["All Workloads"] + all_workloads,
        index=0,
    )

    chart_df = df if selected_workload_filter == "All Workloads" else df[df["resource_id"] == selected_workload_filter]

    # Daily Hours Trend Chart
    st.markdown("#### Daily Compute Usage Hours Trend")
    fig_daily = px.line(
        chart_df,
        x="date",
        y="usage_hours",
        color="resource_id",
        markers=True if selected_workload_filter != "All Workloads" else False,
        labels={"usage_hours": "Usage Hours (0–24h)", "date": "Date", "resource_id": "Workload"},
        title="Daily Compute Hours per Resource",
    )
    fig_daily.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_daily, use_container_width=True)

    # Cost Trend Chart
    st.markdown("#### Daily On-Demand Cost Profile")
    fig_cost = px.bar(
        chart_df,
        x="date",
        y="on_demand_cost",
        color="resource_id",
        barmode="stack",
        labels={"on_demand_cost": "Cost ($ USD)", "date": "Date", "resource_id": "Workload"},
        title="Daily On-Demand Cost Breakdown ($ USD)",
    )
    fig_cost.update_layout(height=350, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_cost, use_container_width=True)

    # Workload Classification: Stable vs Volatile
    st.markdown("#### Workload Stability Classification (Coefficient of Variation)")
    vol_data = []
    for r in recommendations:
        vol_data.append({
            "Workload": r.resource_id,
            "Machine Type": r.machine_type,
            "History (Days)": r.confidence.history_days,
            "Volatility (CV)": r.confidence.coefficient_of_variation,
            "Trend": r.confidence.trend_direction,
            "Confidence": r.confidence.confidence_level,
            "Classification": "Ultra-Stable (24/7)" if r.confidence.coefficient_of_variation < 0.10 else
                             "Moderate Elasticity" if r.confidence.coefficient_of_variation < 0.35 else
                             "High Volatility / Spiky",
        })
    vol_df = pd.DataFrame(vol_data)

    fig_scatter = px.scatter(
        vol_df,
        x="Volatility (CV)",
        y="History (Days)",
        color="Classification",
        text="Workload",
        size=[25] * len(vol_df),
        title="Stability Matrix: Volatility (CV) vs Observation History",
        color_discrete_map={
            "Ultra-Stable (24/7)": "#34A853",
            "Moderate Elasticity": "#4285F4",
            "High Volatility / Spiky": "#EA4335",
        },
    )
    fig_scatter.add_vline(x=0.40, line_dash="dash", line_color="#EA4335", annotation_text="High Volatility Threshold (CV=0.40)")
    fig_scatter.add_hline(y=30, line_dash="dash", line_color="#FBBC04", annotation_text="Minimum History Threshold (30 Days)")
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=380)
    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# TAB 3: DISCOUNT ANALYSIS (SUD vs CUD)
# =========================================================
with tab_discounts:
    st.subheader("Discount Analysis: SUD Credits & CUD Scenarios")

    # SUD Overview Section
    st.markdown("### 1. Sustained Use Discounts (SUD) Performance")
    sud_rows = []
    for s in sud_report.workload_results:
        sud_rows.append({
            "Workload": s.resource_id,
            "Machine Type": s.machine_type,
            "Policy Eligible": "✅ Yes" if s.is_policy_eligible else "❌ No",
            "Eligibility Rationale": s.eligibility_note,
            "Baseline On-Demand": f"${s.total_on_demand_cost:.2f}",
            "Observed SUD Savings": f"${s.observed_sud_savings:.2f}",
            "Effective SUD Rate": f"{s.observed_sud_discount_rate*100:.1f}%",
            "Unrealized Gap": f"${s.unrealized_sud_gap:.2f}",
        })
    st.dataframe(pd.DataFrame(sud_rows), use_container_width=True, hide_index=True)

    if sud_report.policy_overrides:
        st.info("ℹ️ **Policy Enforcement:** " + " | ".join(sud_report.policy_overrides))

    st.markdown("---")

    # CUD Scenarios Comparison
    st.markdown("### 2. Committed Use Discounts (CUD) Portfolio Scenario Comparison")
    st.caption("Normalized Monthly (30-Day) Cost Comparison Across Modeled Strategies")

    scenario_labels = {
        "on_demand_baseline": "100% On-Demand (No Discounts)",
        "current_state": "Current State (Observed SUD + Existing CUD)",
        "cud_conservative": f"Conservative CUD ({int(conserv_cov*100)}% Coverage, 1-Yr)",
        "cud_1yr_full": f"1-Year CUD (100% Coverage, {int(rate_1yr*100)}% Disc.)",
        "cud_3yr_full": f"3-Year CUD (100% Coverage, {int(rate_3yr*100)}% Disc.)",
    }

    scen_plot_data = []
    for s_key, s_label in scenario_labels.items():
        if s_key in cud_report.portfolio_scenarios:
            data = cud_report.portfolio_scenarios[s_key]
            scen_plot_data.append({
                "Scenario": s_label,
                "Monthly Cost ($)": data["monthly_cost"],
                "Monthly Net Savings ($)": data["net_savings"],
            })

    scen_plot_df = pd.DataFrame(scen_plot_data)

    fig_scen = px.bar(
        scen_plot_df,
        x="Scenario",
        y="Monthly Cost ($)",
        color="Scenario",
        text="Monthly Cost ($)",
        title="Portfolio Monthly Cost by Commitment Scenario",
    )
    fig_scen.update_traces(texttemplate="$%{y:,.2f}", textposition="outside")
    fig_scen.update_layout(height=400, showlegend=False, yaxis_title="Monthly Cost ($ USD)")
    st.plotly_chart(fig_scen, use_container_width=True)

    # Opportunity Cost of Displaced SUDs Explainer
    with st.expander("🔍 FinOps Principle: The SUD Displacement Trade-off", expanded=False):
        st.markdown(
            r"""
            **Why CUD savings aren't simply 37% or 55% over current billing:**
            - Eligible Compute Engine VMs running 24/7 already receive up to **30% Sustained Use Discounts automatically** with zero contractual lock-in.
            - When entering a 1-Year CUD (~37% discount), the CUD *replaces* SUD. 
            - The incremental financial gain is **only ~7%** ($37\% - 30\%$), in exchange for accepting **100% utilization liability**.
            - CloudWise explicitly factors this trade-off into every recommendation so you never over-commit for negligible marginal gain.
            """
        )


# =========================================================
# TAB 4: RISK SIMULATOR (STRESS TESTING)
# =========================================================
with tab_risk:
    st.subheader("Commitment Risk & Stress Simulator")
    st.caption("Evaluate downside shelfware exposure when future compute demand declines.")

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("#### Select Workload to Stress Test")
        stress_target_id = st.selectbox(
            "Target Workload:",
            all_workloads,
            index=0,
            key="stress_target_select",
        )
        selected_rec = next(r for r in recommendations if r.resource_id == stress_target_id)

        st.markdown(f"**Machine Type:** `{selected_rec.machine_type}`")
        st.markdown(f"**Recommended Action:** `{selected_rec.recommended_action}`")
        st.markdown(f"**Confidence Level:** `{selected_rec.confidence.confidence_level}`")
        st.markdown(f"**Break-Even Utilization:** `{selected_rec.break_even_utilization_pct}%`")

    with sim_col2:
        st.markdown("#### Stress Multiplier Selector")
        interactive_mult = st.select_slider(
            "Simulate Future Usage Level:",
            options=[0.60, 0.75, 0.90, 1.00, 1.20],
            value=0.75,
            format_func=lambda x: f"{int(x*100)}% ({'Expected Base' if x==1.0 else f'{int((x-1)*100)}% Shift'})",
        )

    # Render stress-testing table and chart
    stress_rows = []
    for s in selected_rec.stress_test_results:
        stress_rows.append({
            "Usage Level": s.label,
            "Projected Hours": f"{s.projected_usage_hours:.1f}h",
            "Utilization (%)": f"{s.commitment_utilization_pct:.1f}%",
            "Unused Commitment (Shelfware)": f"${s.unused_commitment_exposure:.2f}",
            "Scenario Cost ($)": f"${s.scenario_total_cost:.2f}",
            "Net Savings ($)": f"${s.net_savings_vs_baseline:+.2f}",
            "Financially Beneficial": "✅ Yes" if s.is_financially_beneficial else "❌ No (Loss)",
        })

    st.markdown("#### Stress Scenarios Matrix (60% to 120%)")
    st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

    # Plotly Stress Chart
    stress_chart_df = pd.DataFrame([
        {
            "Usage Level": s.label,
            "Multiplier": s.multiplier,
            "Net Savings ($)": s.net_savings_vs_baseline,
            "Unused Commitment Shelfware ($)": s.unused_commitment_exposure,
            "Utilization (%)": s.commitment_utilization_pct,
        }
        for s in selected_rec.stress_test_results
    ])

    fig_stress = go.Figure()
    fig_stress.add_trace(go.Bar(
        x=stress_chart_df["Usage Level"],
        y=stress_chart_df["Net Savings ($)"],
        name="Net Financial Savings ($)",
        marker_color=["#34A853" if v >= 0 else "#EA4335" for v in stress_chart_df["Net Savings ($)"]],
    ))
    fig_stress.add_trace(go.Scatter(
        x=stress_chart_df["Usage Level"],
        y=stress_chart_df["Unused Commitment Shelfware ($)"],
        name="Unused Commitment Waste ($)",
        mode="lines+markers",
        line=dict(color="#F9AB00", width=3, dash="dot"),
    ))
    fig_stress.add_hline(y=0, line_dash="solid", line_color="#202124", line_width=1)
    fig_stress.update_layout(
        title=f"Downside Risk Profile for '{stress_target_id}' Across Demand Drops",
        height=380,
        yaxis_title="Dollars ($ USD)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    # Highlighting selected interactive multiplier
    matched_sim = next(s for s in selected_rec.stress_test_results if abs(s.multiplier - interactive_mult) < 0.01)
    st.markdown(
        f'<div class="warning-banner">'
        f'<strong>Analysis at {interactive_mult*100:.0f}% Demand:</strong> '
        f'{matched_sim.explanation}'
        f'</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# TAB 5: RECOMMENDATIONS & REPORT EXPORT
# =========================================================
with tab_recs:
    st.subheader("FinOps Recommendation Report")
    st.caption("Actionable, auditable recommendations with confidence scores and justifications.")

    # Download Button
    csv_report_data = export_recommendations_to_csv(recommendations)
    st.download_button(
        label="📥 Download Full Recommendations Report (CSV)",
        data=csv_report_data,
        file_name="cloudwise_finops_recommendations.csv",
        mime="text/csv",
        help="Download auditable CSV report formatted for stakeholder presentation.",
    )

    st.markdown("---")

    # Render Cards per Workload
    for r in recommendations:
        with st.container():
            rec_header_col, rec_badge_col, rec_metric_col = st.columns([3, 1.5, 2])

            with rec_header_col:
                st.markdown(f"### 🖥️ `{r.resource_id}` ({r.machine_type})")
                st.markdown(f"**Project:** `{r.project_id}`")

            with rec_badge_col:
                if r.action_type == "COMMIT":
                    st.markdown(f'<span class="badge-commit">{r.recommended_action}</span>', unsafe_allow_html=True)
                elif r.action_type == "WAIT":
                    st.markdown(f'<span class="badge-wait">{r.recommended_action}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-flexible">{r.recommended_action}</span>', unsafe_allow_html=True)

                st.caption(f"Confidence: **{r.confidence.confidence_level}** ({r.confidence.overall_score}/100 pts)")

            with rec_metric_col:
                if r.estimated_monthly_savings > 0:
                    st.metric("Est. Monthly Savings", f"${r.estimated_monthly_savings:.2f}", f"+${r.estimated_annual_savings:.2f}/yr")
                else:
                    st.metric("Est. Incremental Savings", "$0.00", "Maintain Current Posture")

            st.markdown(f"**Rationale:** {r.primary_reason}")
            if r.downside_risk_warning:
                st.caption(f"⚠️ *Downside Risk:* {r.downside_risk_warning}")

            with st.expander("Confidence Breakdown & Assumptions", expanded=False):
                st.markdown("**Confidence Rubric Points:**")
                for bullet in r.confidence.rationale_bullets:
                    st.markdown(f"- {bullet}")
                st.markdown("**Key Assumptions:**")
                for asm in r.assumptions_and_limitations:
                    st.markdown(f"- {asm}")

            st.divider()

    # Full tabular dataframe view
    st.markdown("#### Tabular Summary")
    st.dataframe(recs_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 6: METHODOLOGY & AUDIT LOG
# =========================================================
with tab_methodology:
    st.subheader("FinOps Engine Architecture & Mathematical Formulations")

    st.markdown(
        r"""
        ### 📐 Calculation Formulas & Assumptions

        #### 1. Baseline On-Demand Equivalence
        $$OD_{total} = \\sum_{t} (\\text{usage\\_hours}_t \\times \\text{hourly\\_rate})$$

        #### 2. Fixed-Price CUD Formula
        For a fixed-price commitment on newly committed capacity $A$ with discount $d$:
        $$C_{commit} = A \\times (1 - d)$$
        $$\\text{Net Savings} = A - C_{commit} - \\text{SUD}_{\\text{displaced}}$$
        *Applies when committed instance runs continuously and pricing remains static.*

        #### 3. Google Cloud Sustained Use Discount (SUD) Curve
        Applied to eligible machine families (N1, N2, N2D, C2, M1) based on monthly run fraction $f$:
        - $0\\% - 25\\%$ of month: $0\\%$ discount
        - $25\\% - 50\\%$ of month: $20\\%$ incremental discount
        - $50\\% - 75\\%$ of month: $40\\%$ incremental discount
        - $75\\% - 100\\%$ of month: $60\\%$ incremental discount
        - Max blended discount for $100\\%$ continuous runtime: **$30\\%$ net discount**.

        #### 4. Break-Even Utilization
        $$\\text{Break-Even Utilization} = 1.0 - d$$
        - 1-Year CUD ($37\\%$ discount): **$63.0\\%$ utilization required**.
        - 3-Year CUD ($55\\%$ discount): **$45.0\\%$ utilization required**.
        - Operating below break-even generates pure shelfware loss.

        #### 5. Ground-Truth History vs Blind Trust
        Rather than trusting user-provided metadata, CloudWise computes:
        $$\\text{History Days} = (\\text{Date}_{\\max} - \\text{Date}_{\\min}) + 1$$
        $$\\text{Completeness} = \\frac{\\text{Distinct Recorded Dates}}{\\text{Calendar Span}}$$
        """
    )

    st.markdown("---")
    st.markdown("### 🔌 Real Google Cloud BigQuery Integration Specification")
    st.markdown(
        """
        To transition from this MVP to live Google Cloud Billing export telemetry, the following schema mapping is verified:
        - **BigQuery Export Table:** `[project].[dataset].gcp_billing_export_resource_v1_[billing_account]` (Detailed Resource Export is required for `resource.name`).
        - **Date Dimension:** Use `DATE(usage_start_time, 'UTC')` for true consumption date (avoid `_PARTITIONTIME` due to export ingestion latency).
        - **Resource Identifier:** `COALESCE(resource.name, resource.global_name)`.
        - **Machine Spec / Type Attribution:** Read from `(SELECT value FROM UNNEST(system_labels) WHERE key = 'compute.googleapis.com/machine_spec')` or correlate with Cloud Asset Inventory.
        - **Runtime Hours:** Filter to vCPU core hours where `usage.pricing_unit IN ('hour', 'seconds')` (never sum raw `amount_in_pricing_units` across mixed RAM/Disk SKUs).
        - **Cost Baseline:** Gross `cost` prior to credits.
        - **Credits Array:** Sum `ABS(c.amount)` for `credits.type = 'SUSTAINED_USAGE_DISCOUNT'` and `'COMMITTED_USAGE_DISCOUNT'`.
        """
    )
