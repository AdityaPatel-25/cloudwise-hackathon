"""
Report Generator Module for CloudWise.

Compiles workload recommendations, financial savings, risk assessments,
and confidence metrics into tabular formats and downloadable CSV exports.
"""

from typing import Dict, List, Optional
import pandas as pd
from .risk_engine import WorkloadRecommendation


def build_recommendations_dataframe(recs: List[WorkloadRecommendation]) -> pd.DataFrame:
    """
    Convert a list of WorkloadRecommendation objects into a clean, display-ready
    and exportable pandas DataFrame.
    """
    rows = []
    for r in recs:
        rows.append({
            "Workload ID": r.resource_id,
            "Project ID": r.project_id,
            "Machine Type": r.machine_type,
            "Recommended Action": r.recommended_action,
            "Action Category": r.action_type,
            "Confidence Level": r.confidence.confidence_level,
            "Confidence Score": r.confidence.overall_score,
            "Observed History (Days)": r.confidence.history_days,
            "Usage Volatility (CV)": round(r.confidence.coefficient_of_variation, 2),
            "Usage Trend": r.confidence.trend_direction,
            "Estimated Monthly Savings ($)": round(r.estimated_monthly_savings, 2),
            "Estimated Annual Savings ($)": round(r.estimated_annual_savings, 2),
            "Break-Even Utilization (%)": r.break_even_utilization_pct,
            "Primary Justification": r.primary_reason,
            "Downside Risk Warning": r.downside_risk_warning,
            "Key Assumptions": "; ".join(r.assumptions_and_limitations),
        })

    df = pd.DataFrame(rows)
    # Sort with highest savings first, then by workload
    if not df.empty and "Estimated Monthly Savings ($)" in df.columns:
        df = df.sort_values(by=["Estimated Monthly Savings ($)", "Workload ID"], ascending=[False, True]).reset_index(drop=True)
    return df


def generate_executive_summary_kpis(
    df: pd.DataFrame,
    recs: List[WorkloadRecommendation],
    cud_portfolio: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate high-level FinOps executive KPIs for dashboard cards.
    """
    total_on_demand_modeled = float(df["on_demand_cost"].sum())
    total_workloads = int(df["resource_id"].nunique())

    total_monthly_est_savings = sum(max(0.0, float(r.estimated_monthly_savings)) for r in recs)
    total_annual_est_savings = sum(max(0.0, float(r.estimated_annual_savings)) for r in recs)

    actions_breakdown = {}
    confidence_breakdown = {}
    for r in recs:
        actions_breakdown[r.action_type] = actions_breakdown.get(r.action_type, 0) + 1
        confidence_breakdown[r.confidence.confidence_level] = confidence_breakdown.get(r.confidence.confidence_level, 0) + 1

    return {
        "total_on_demand_modeled": round(total_on_demand_modeled, 2),
        "total_workloads_analyzed": total_workloads,
        "total_monthly_incremental_savings": round(total_monthly_est_savings, 2),
        "total_annual_incremental_savings": round(total_annual_est_savings, 2),
        "actions_breakdown": actions_breakdown,
        "confidence_breakdown": confidence_breakdown,
    }


def export_recommendations_to_csv(recs: List[WorkloadRecommendation]) -> str:
    """Return CSV formatted string of recommendations for download."""
    df = build_recommendations_dataframe(recs)
    return df.to_csv(index=False)
