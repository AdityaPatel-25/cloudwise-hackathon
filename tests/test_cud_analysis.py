"""
Unit tests for src/cud_analysis.py and src/sud_analysis.py.

Verifies hand-calculable scenarios, existing commitment coverage,
discount double-counting prevention, and break-even calculations.
"""

from io import StringIO
import pytest
import pandas as pd
from src.data_loader import load_usage_data
from src.sud_analysis import (
    analyze_sud,
    calculate_tiered_sud_discount_factor,
    extract_machine_family,
)
from src.cud_analysis import (
    evaluate_workload_cud_scenarios,
    analyze_cud_portfolio,
    DEFAULT_CUD_RATES,
)


def create_mock_workload_df(
    resource_id: str = "test-vm",
    machine_type: str = "n2-standard-4",
    num_days: int = 30,
    daily_hours: float = 24.0,
    daily_cost: float = 10.0,
    existing_commitment: float = 0.0,
    sud_eligible: bool = True,
) -> pd.DataFrame:
    rows = []
    for day in range(1, num_days + 1):
        d_str = f"2026-07-{day:02d}"
        rows.append({
            "date": d_str,
            "project_id": "test-project",
            "resource_id": resource_id,
            "machine_type": machine_type,
            "region": "us-central1",
            "usage_hours": daily_hours,
            "on_demand_cost": daily_cost,
            "existing_commitment": existing_commitment,
            "sud_eligible": sud_eligible,
            "calculated_history_days": num_days,
            "data_completeness_pct": 100.0,
        })
    return pd.DataFrame(rows)


def test_hand_calculable_cud_baseline_and_savings():
    """
    Hand-calculable test:
    30 days, $10.00/day on-demand => $300.00 total on-demand.
    0% existing commitment. Machine is e2 (no SUD to isolate CUD math).
    1-Year CUD (37% discount):
      Commitment fee C = 300 * (1 - 0.37) = $189.00
      Net savings = 300 - 189 = $111.00
      Break-even utilization = 100 - 37 = 63.0%
    3-Year CUD (55% discount):
      Commitment fee C = 300 * (1 - 0.55) = $135.00
      Net savings = 300 - 135 = $165.00
      Break-even utilization = 100 - 55 = 45.0%
    """
    df = create_mock_workload_df(
        machine_type="e2-standard-4",
        num_days=30,
        daily_hours=24.0,
        daily_cost=10.0,
        existing_commitment=0.0,
        sud_eligible=False,
    )

    res = evaluate_workload_cud_scenarios(
        workload_df=df,
        rates={"cud_1yr": 0.37, "cud_3yr": 0.55, "conservative_coverage": 0.70},
        enforce_gcp_family_policy=True,
    )

    assert res.total_on_demand_baseline == 300.00

    # 1-Year Full CUD
    s_1yr = res.scenarios["cud_1yr_full"]
    assert s_1yr.period_commitment_fee == 189.00
    assert s_1yr.period_total_cost == 189.00
    assert s_1yr.period_net_savings_vs_current_state == 111.00
    assert s_1yr.break_even_utilization_pct == 63.0

    # 3-Year Full CUD
    s_3yr = res.scenarios["cud_3yr_full"]
    assert s_3yr.period_commitment_fee == 135.00
    assert s_3yr.period_total_cost == 135.00
    assert s_3yr.period_net_savings_vs_current_state == 165.00
    assert s_3yr.break_even_utilization_pct == 45.0

    # Conservative CUD (70% coverage at 37% rate)
    # Committed portion = 300 * 0.70 = 210.00
    # Commit fee = 210 * 0.63 = 132.30
    # Uncovered portion = 300 * 0.30 = 90.00
    # Total cost = 132.30 + 90.00 = 222.30
    # Net savings = 300 - 222.30 = 77.70
    s_conserv = res.scenarios["cud_conservative"]
    assert s_conserv.period_commitment_fee == 132.30
    assert s_conserv.period_total_cost == 222.30
    assert s_conserv.period_net_savings_vs_current_state == 77.70


def test_existing_commitment_coverage_accounting():
    """
    Test workload with 50% existing commitment coverage:
    Total OD = $300.
    50% existing => $150 already covered at existing 1-year rate ($150 * 0.63 = $94.50).
    Only remaining $150 is evaluated for incremental commitment.
    New 1-Yr CUD on remaining $150: fee = 150 * 0.63 = $94.50.
    Total cost with new CUD = 94.50 + 94.50 = $189.00.
    Current state cost = 94.50 (existing commit) + 150.00 (uncovered OD) = $244.50.
    Incremental Net savings = 244.50 - 189.00 = $55.50.
    """
    df = create_mock_workload_df(
        machine_type="e2-standard-4",
        num_days=30,
        daily_hours=24.0,
        daily_cost=10.0,
        existing_commitment=0.50,
        sud_eligible=False,
    )

    res = evaluate_workload_cud_scenarios(
        workload_df=df,
        rates={"cud_1yr": 0.37, "cud_3yr": 0.55, "conservative_coverage": 0.70},
    )

    assert res.existing_commitment_ratio == 0.50
    assert res.existing_commitment_cost == 94.50
    assert res.uncovered_on_demand_cost == 150.00
    assert res.current_state_cost == 244.50

    s_1yr = res.scenarios["cud_1yr_full"]
    assert s_1yr.period_total_cost == 189.00
    assert s_1yr.period_net_savings_vs_current_state == 55.50


def test_sud_tiered_discount_factor():
    """Verify Google Cloud tiered SUD calculation curve."""
    # 0 hours -> 0%
    assert calculate_tiered_sud_discount_factor(0.0) == 0.0

    # Running <= 25% of the month -> 0% discount
    assert calculate_tiered_sud_discount_factor(0.20) == 0.0
    assert calculate_tiered_sud_discount_factor(0.25) == 0.0

    # Running 50% of the month:
    # First 25% @ 1.0, second 25% @ 0.8 => avg cost ratio = (0.25*1 + 0.25*0.8)/0.50 = 0.45/0.50 = 0.90
    # Discount = 1.0 - 0.90 = 10% (0.10)
    assert calculate_tiered_sud_discount_factor(0.50) == 0.10

    # Running 100% of the month (full 730h month):
    # Tier 1 (25% @ 1.0) = 0.25
    # Tier 2 (25% @ 0.8) = 0.20
    # Tier 3 (25% @ 0.6) = 0.15
    # Tier 4 (25% @ 0.4) = 0.10
    # Sum = 0.70 => Net discount = 30% (0.30)
    assert calculate_tiered_sud_discount_factor(1.00) == 0.30


def test_machine_family_ineligibility_override():
    """Verify that E2 and C3 machine types are recognized as SUD ineligible."""
    assert extract_machine_family("e2-standard-4") == "e2"
    assert extract_machine_family("n2-standard-8") == "n2"
    assert extract_machine_family("c3-standard-4") == "c3"

    df_e2 = create_mock_workload_df(
        machine_type="e2-standard-4",
        num_days=30,
        daily_hours=24.0,
        daily_cost=10.0,
        sud_eligible=True,  # Synthetic data incorrectly said True
    )

    sud_rep = analyze_sud(df_e2, enforce_gcp_family_policy=True)
    w_res = sud_rep.workload_results[0]
    assert w_res.is_policy_eligible is False
    assert w_res.observed_sud_savings == 0.0
    assert len(sud_rep.policy_overrides) > 0


def test_prevent_discount_double_counting():
    """
    Verify that an instance covered by 100% existing commitment does NOT receive SUD
    on those same covered hours.
    """
    df_100_commit = create_mock_workload_df(
        machine_type="n2-standard-4",
        num_days=30,
        daily_hours=24.0,
        daily_cost=10.0,
        existing_commitment=1.00,
        sud_eligible=True,
    )

    sud_rep = analyze_sud(df_100_commit, enforce_gcp_family_policy=True)
    w_res = sud_rep.workload_results[0]
    # Uncovered ratio is 0.0 => observed SUD must be 0.0
    assert w_res.uncovered_usage_ratio == 0.0
    assert w_res.observed_sud_savings == 0.0


def test_reconciliation_exact_tiered_sud_vs_approximate_savings():
    """
    Reconciliation Regression Test:
    Explains the exact arithmetic difference between the $69.89/mo reported savings
    and the simplified approximation $69.55/mo:

    Simplified Approximation:
      Assumes flat 100% runtime -> flat 30.00% SUD credit.
      Monthly OD: $278.22
      Displaced SUD: $278.22 * 0.30 = $83.47/mo
      3-Yr CUD gross: $278.22 * 0.55 = $153.02/mo
      Approx Net Savings = $153.02 - $83.47 = $69.55/mo

    Exact Implementation in CloudWise:
      Workload wl-prod-db-01 actually runs 23.902 hrs/day (~99.59% runtime).
      Tiered SUD curve yields effective discount of 29.88% (not 30.00%).
      90-day OD cost: $834.66
      90-day SUD credit: round(834.66 * 0.2988, 2) = $249.40 ($83.13/mo)
      90-day 3-Yr CUD cost: round(834.66 * (1 - 0.55), 2) = $375.60 ($125.20/mo)
      90-day Net Savings = (834.66 - 249.40) - 375.60 = 585.26 - 375.60 = $209.66
      Monthly Net Savings = round(209.66 * (30 / 90), 2) = round(69.8867, 2) = $69.89/mo!

      The $0.34/mo delta ($69.89 - $69.55) is exactly the $0.34 difference between
      the hypothetical flat 30% SUD ($83.47) and the actual tiered 29.88% SUD ($83.13).
    """
    from src.data_loader import load_usage_data, get_sample_csv_path

    df = load_usage_data(get_sample_csv_path()).df
    db_df = df[df["resource_id"] == "wl-prod-db-01"]

    res = evaluate_workload_cud_scenarios(db_df)
    s_3yr = res.scenarios["cud_3yr_full"]

    # Verify baseline and runtime
    assert round(res.total_on_demand_baseline, 2) == 834.66
    assert res.period_days == 90

    # Verify exact tiered SUD rate on 99.59% runtime
    total_hours = db_df["usage_hours"].sum()
    run_fraction = (total_hours / 90.0) / 24.0
    assert 0.995 < run_fraction < 1.0
    exact_sud_rate = calculate_tiered_sud_discount_factor(run_fraction)
    assert exact_sud_rate == 0.2988  # 29.88%

    # Verify exact 90-day SUD credit and net savings
    assert res.current_state_sud_credit == 249.40
    assert s_3yr.period_net_savings_vs_current_state == 209.66
    assert s_3yr.monthly_net_savings == 69.89

    # Now verify: If workload ran 24.0h flat (100% of the month), it produces $69.55!
    flat_df = create_mock_workload_df(
        resource_id="flat-db-01",
        machine_type="n2-standard-8",
        num_days=90,
        daily_hours=24.0,
        daily_cost=round(834.66 / 90.0, 4),  # ~$9.274/day
        existing_commitment=0.0,
        sud_eligible=True,
    )
    flat_res = evaluate_workload_cud_scenarios(flat_df)
    flat_3yr = flat_res.scenarios["cud_3yr_full"]
    # At exactly 100.0% runtime, SUD rate is exactly 0.3000
    assert flat_res.current_state_sud_credit == round(flat_res.total_on_demand_baseline * 0.30, 2)
    # And monthly net savings matches approximately $69.55
    assert abs(flat_3yr.monthly_net_savings - 69.55) <= 0.05

