"""
Unit tests for src/risk_engine.py.

Verifies stress simulation (60%-120%), unused commitment exposure,
confidence rubric, short-history WAIT recommendation, declining workload handling,
and deterministic outputs.
"""

import pytest
import pandas as pd
import numpy as np
from src.risk_engine import (
    simulate_commitment_stress,
    evaluate_confidence,
    generate_recommendation_for_workload,
    generate_portfolio_recommendations,
)
from src.data_loader import load_usage_data, get_sample_csv_path


def create_workload(
    resource_id: str,
    days_data: list,
    machine_type: str = "n2-standard-4",
    existing_commit: float = 0.0,
    sud_eligible: bool = True,
) -> pd.DataFrame:
    rows = []
    num_days = len(days_data)
    for i, h in enumerate(days_data):
        d_str = f"2026-07-{(i+1):02d}"
        rows.append({
            "date": d_str,
            "project_id": "prj-test",
            "resource_id": resource_id,
            "machine_type": machine_type,
            "region": "us-central1",
            "usage_hours": float(h),
            "on_demand_cost": round(float(h) * 0.20, 4),
            "existing_commitment": existing_commit,
            "sud_eligible": sud_eligible,
            "calculated_history_days": num_days,
            "active_recorded_days": num_days,
            "data_completeness_pct": 100.0,
        })
    return pd.DataFrame(rows)


def test_stress_testing_underutilization_and_shelfware():
    """
    Test that at 60% usage drop:
    - Commitment utilization is ~60%
    - Unused commitment exposure is positive (wasted money)
    - Financial net savings can turn negative.
    """
    stress_results = simulate_commitment_stress(
        baseline_uncovered_od=1000.0,
        target_coverage_ratio=1.0,
        discount_rate=0.37,  # 1-Year CUD
        is_sud_eligible=False,
        avg_daily_hours=24.0,
        existing_commit_fee=0.0,
        period_days=30,
    )

    assert len(stress_results) == 5
    res_60 = next(r for r in stress_results if r.multiplier == 0.60)
    res_100 = next(r for r in stress_results if r.multiplier == 1.00)

    # At 100% expected usage:
    # 100% utilized, 0 unused exposure, net savings > 0
    assert res_100.commitment_utilization_pct == 100.0
    assert res_100.unused_commitment_exposure == 0.0
    assert res_100.net_savings_vs_baseline > 0
    assert res_100.is_financially_beneficial is True

    # At 60% usage:
    # Fixed commitment fee is 1000 * 0.63 = $630.00
    # Future baseline demand at 60% is 1000 * 0.60 = $600.00
    # Utilization = 60.0%
    # Unused shelfware exposure = 40% of $630 = $252.00
    # Scenario cost = $630.00, Baseline cost = $600.00
    # Net savings = 600 - 630 = -$30.00 (loss!)
    assert res_60.commitment_utilization_pct == 60.0
    assert res_60.unused_commitment_exposure == 252.00
    assert res_60.net_savings_vs_baseline == -30.00
    assert res_60.is_financially_beneficial is False


def test_short_history_triggers_wait_recommendation():
    """
    Workload with only 15 days of data must produce:
    - Recommended action: "Wait (Insufficient History)"
    - Action type: "WAIT"
    - Confidence: "Insufficient Evidence"
    """
    short_days = [24.0] * 15
    df_short = create_workload(resource_id="pilot-vm", days_data=short_days)

    rec = generate_recommendation_for_workload(df_short)

    assert rec.action_type == "WAIT"
    assert "Wait" in rec.recommended_action
    assert rec.confidence.confidence_level == "Insufficient Evidence"
    assert rec.estimated_monthly_savings == 0.0
    assert "below the required 30-day" in rec.primary_reason


def test_declining_workload_triggers_do_not_commit():
    """
    Workload with sharply declining usage over 60 days:
    Days 1-30: 24 hrs/day
    Days 31-60: 4 hrs/day
    Must produce:
    - Action category: "FLEXIBLE"
    - "Do Not Commit" or "Declining" in recommendation
    """
    declining_days = ([24.0] * 30) + ([4.0] * 30)
    df_decline = create_workload(resource_id="decommissioning-vm", days_data=declining_days)

    rec = generate_recommendation_for_workload(df_decline)

    assert rec.action_type == "FLEXIBLE"
    assert "Declining" in rec.recommended_action or "Do Not Commit" in rec.recommended_action
    assert "declining" in rec.primary_reason.lower()


def test_bursty_workload_triggers_flexibility():
    """
    Workload with high CV (>0.40) due to sporadic spikes:
    Must produce:
    - Action category: "FLEXIBLE"
    - Recommends maintaining flexibility / SUD
    """
    # 25 days of 2 hrs, 5 days of 24 hrs => high variance
    bursty_days = ([2.0] * 25) + ([24.0] * 5)
    df_burst = create_workload(resource_id="spiky-worker", days_data=bursty_days)

    rec = generate_recommendation_for_workload(df_burst)

    assert rec.action_type == "FLEXIBLE"
    assert rec.confidence.coefficient_of_variation > 0.40
    assert "Flexibility" in rec.recommended_action or "Shutdown" in rec.recommended_action


def test_stable_production_triggers_commitment():
    """
    Workload running 24 hrs flat for 90 days with 0% existing coverage:
    Must produce:
    - Action category: "COMMIT"
    - High Confidence
    - Positive estimated monthly savings
    """
    stable_days = [24.0] * 90
    df_stable = create_workload(resource_id="prod-db", days_data=stable_days, machine_type="n2-standard-8")

    rec = generate_recommendation_for_workload(df_stable)

    assert rec.action_type == "COMMIT"
    assert rec.confidence.confidence_level in ["High", "Medium"]
    assert rec.estimated_monthly_savings > 0.0
    assert rec.estimated_annual_savings > 0.0


def test_deterministic_reproducible_recommendations():
    """
    Running recommendation generation twice on the bundled sample dataset
    must produce identical outputs.
    """
    path = get_sample_csv_path()
    res = load_usage_data(path)
    assert res.is_valid is True

    recs_run1 = generate_portfolio_recommendations(res.df)
    recs_run2 = generate_portfolio_recommendations(res.df)

    assert len(recs_run1) == len(recs_run2)
    for r1, r2 in zip(recs_run1, recs_run2):
        assert r1.resource_id == r2.resource_id
        assert r1.recommended_action == r2.recommended_action
        assert r1.estimated_monthly_savings == r2.estimated_monthly_savings
        assert r1.confidence.overall_score == r2.confidence.overall_score


def test_zero_discount_behavior_produces_zero_savings_and_no_commitments():
    """
    When CUD discount rates are 0% (or 0% coverage), zero incremental savings
    must be reported, and NO workload should be recommended for a COMMIT action.
    """
    from src.report_generator import generate_executive_summary_kpis

    path = get_sample_csv_path()
    res = load_usage_data(path)
    assert res.is_valid is True

    rates_zero = {"cud_1yr": 0.0, "cud_3yr": 0.0, "conservative_coverage": 0.0}
    recs = generate_portfolio_recommendations(res.df, rates=rates_zero)
    kpis = generate_executive_summary_kpis(res.df, recs)

    # Must be strictly 0.0 incremental savings
    assert kpis["total_monthly_incremental_savings"] == 0.0
    assert kpis["total_annual_incremental_savings"] == 0.0

    # No workload can have a COMMIT action category
    assert kpis["actions_breakdown"].get("COMMIT", 0) == 0
    for r in recs:
        assert r.action_type in ["FLEXIBLE", "WAIT", "NO_CHANGE"]
        assert r.estimated_monthly_savings == 0.0
        assert r.estimated_annual_savings == 0.0


def test_positive_discount_behavior_hand_calculable():
    """
    Hand-calculable positive discount test:
    Single stable workload (30 days, 24h/day, $10/day = $300 OD).
    Non-SUD eligible (e2-standard-4) so CUD savings is pure (1 - d).
    3-Year CUD (55% discount):
      Monthly savings = $300 * 0.55 = $165.00/month.
      Annual savings = $165.00 * (365/30) = $2007.50/year.
    """
    stable_days = [24.0] * 30
    df = create_workload(resource_id="hand-calc-vm", days_data=stable_days, machine_type="e2-standard-4")
    # Set on_demand_cost = $10.00/day
    df["on_demand_cost"] = 10.00

    rates = {"cud_1yr": 0.37, "cud_3yr": 0.55, "conservative_coverage": 0.70}
    rec = generate_recommendation_for_workload(df, rates=rates)

    assert rec.action_type == "COMMIT"
    assert "3-Year CUD" in rec.recommended_action
    assert rec.estimated_monthly_savings == 165.00
    assert rec.estimated_annual_savings == 2007.50


def test_zero_conservative_coverage_behavior():
    """
    When conservative target coverage is 0%, conservative scenario commits 0% and saves $0.
    If 1-year CUD full (100%) produces savings, it should recommend 1-Year CUD full,
    NOT Conservative CUD with 0% coverage.
    """
    stable_days = [24.0] * 30
    df = create_workload(resource_id="zero-cov-vm", days_data=stable_days, machine_type="e2-standard-4")
    df["on_demand_cost"] = 10.00

    # 1-yr positive (37%), 3-yr zero (0%), conservative coverage zero (0%)
    rates = {"cud_1yr": 0.37, "cud_3yr": 0.0, "conservative_coverage": 0.0}
    rec = generate_recommendation_for_workload(df, rates=rates)

    assert rec.action_type == "COMMIT"
    assert "1-Year CUD (100% Coverage)" in rec.recommended_action
    assert rec.estimated_monthly_savings == 111.00  # $300 * 0.37 = $111


def test_existing_commitment_coverage_isolation():
    """
    Workload with 50% existing commitment:
    Only remaining 50% uncovered is evaluated for new commitments.
    Changing future candidate rate does NOT change past existing commitment cost.
    """
    stable_days = [24.0] * 30
    df_half = create_workload(
        resource_id="half-commit-vm",
        days_data=stable_days,
        machine_type="e2-standard-4",
        existing_commit=0.50,
    )
    df_half["on_demand_cost"] = 10.00  # $300 total

    # At 0% future rates:
    rates_zero = {"cud_1yr": 0.0, "cud_3yr": 0.0, "conservative_coverage": 0.0}
    rec_zero = generate_recommendation_for_workload(df_half, rates=rates_zero)
    assert rec_zero.action_type == "FLEXIBLE"
    assert rec_zero.estimated_monthly_savings == 0.0

    # At 37% 1-year rate and 70% conservative coverage on remaining $150 uncovered:
    # Newly committed = $150 * 0.70 = $105. Net savings = $105 * 0.37 = $38.85
    rates_pos = {"cud_1yr": 0.37, "cud_3yr": 0.0, "conservative_coverage": 0.70}
    rec_pos = generate_recommendation_for_workload(df_half, rates=rates_pos)
    assert rec_pos.action_type == "COMMIT"
    assert rec_pos.estimated_monthly_savings == 38.85


def test_annual_savings_calculations_consistency():
    """
    Verify annual savings scaling consistency across the report generator.
    """
    from src.report_generator import build_recommendations_dataframe, generate_executive_summary_kpis

    path = get_sample_csv_path()
    res = load_usage_data(path)
    assert res.is_valid is True

    recs = generate_portfolio_recommendations(res.df)
    kpis = generate_executive_summary_kpis(res.df, recs)
    df_report = build_recommendations_dataframe(recs)

    # Check sum in report dataframe matches KPI card
    sum_monthly_df = df_report["Estimated Monthly Savings ($)"].sum()
    sum_annual_df = df_report["Estimated Annual Savings ($)"].sum()

    assert round(sum_monthly_df, 2) == kpis["total_monthly_incremental_savings"]
    assert round(sum_annual_df, 2) == kpis["total_annual_incremental_savings"]


def test_recommendation_counts_match_dashboard_data():
    """
    Verify that actions_breakdown in summary KPIs sums to total unique workloads
    and matches individual recommendation action types.
    """
    from src.report_generator import generate_executive_summary_kpis

    path = get_sample_csv_path()
    res = load_usage_data(path)
    assert res.is_valid is True

    recs = generate_portfolio_recommendations(res.df)
    kpis = generate_executive_summary_kpis(res.df, recs)

    total_actions_count = sum(kpis["actions_breakdown"].values())
    assert total_actions_count == kpis["total_workloads_analyzed"]
    assert total_actions_count == len(recs)

