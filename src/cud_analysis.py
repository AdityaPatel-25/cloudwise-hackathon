"""
Committed Use Discount (CUD) Scenario Analysis Engine for CloudWise.

Evaluates Google Cloud Committed Use Discounts against on-demand and SUD baselines.
Models:
1. Pure On-Demand Baseline
2. Current State (accounting for existing commitments & active SUDs)
3. 1-Year CUD Scenario (illustrative ~37% discount)
4. 3-Year CUD Scenario (illustrative ~55% discount)
5. Conservative CUD Scenario (e.g. 70% coverage at 1-year rate)
6. Flexible Strategy (No New Commitment, rely on SUD / elasticity)

Enforces FinOps best practices:
- Distinguishes baseline, commitment fee, covered usage value, and uncovered usage.
- Models the "SUD Trade-off" (incremental gain over existing SUD vs flexibility loss).
- Prevents double-counting discounts across overlapping commitment windows.
- Normalizes comparisons to evaluated period, monthly (30-day), and annual bases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .sud_analysis import GCP_SUD_FAMILY_POLICY, extract_machine_family, calculate_tiered_sud_discount_factor


# Default illustrative CUD discount rates (clearly configurable and labelled illustrative)
DEFAULT_CUD_RATES = {
    "cud_1yr": 0.37,          # Illustrative 1-Year CUD rate (~37% discount)
    "cud_3yr": 0.55,          # Illustrative 3-Year CUD rate (~55% discount)
    "conservative_coverage": 0.70,  # Recommended conservative commitment portion (70%)
}


@dataclass
class ScenarioMetrics:
    """Financial metrics for a specific commitment scenario for a workload."""
    scenario_id: str
    scenario_name: str
    commitment_term: str  # "None", "1-Year", "3-Year"
    target_coverage_ratio: float  # newly covered fraction of uncovered baseline
    effective_discount_rate: float
    # Costs over evaluated dataset period
    period_commitment_fee: float
    period_uncovered_cost: float
    period_sud_credits_earned: float
    period_total_cost: float
    period_gross_savings_vs_ondemand: float
    period_net_savings_vs_current_state: float
    # Normalized 30-day monthly values
    monthly_total_cost: float
    monthly_net_savings: float
    # Normalized 365-day annual values
    annual_total_cost: float
    annual_net_savings: float
    # Risk indicators
    break_even_utilization_pct: float
    assumptions: str


@dataclass
class WorkloadCUDResult:
    """Full scenario suite for a single workload."""
    resource_id: str
    project_id: str
    machine_type: str
    machine_family: str
    is_sud_eligible: bool
    history_days: int
    period_days: int
    existing_commitment_ratio: float
    total_on_demand_baseline: float
    existing_commitment_cost: float
    uncovered_on_demand_cost: float
    current_state_cost: float
    current_state_sud_credit: float
    scenarios: Dict[str, ScenarioMetrics] = field(default_factory=dict)
    best_financial_scenario: str = "no_new_commit"


@dataclass
class PortfolioCUDReport:
    """Aggregated portfolio-level CUD scenario comparison."""
    workload_results: List[WorkloadCUDResult]
    portfolio_scenarios: Dict[str, Dict[str, float]]
    configured_rates: Dict[str, float]
    methodology_notes: List[str]


def evaluate_workload_cud_scenarios(
    workload_df: pd.DataFrame,
    rates: Optional[Dict[str, float]] = None,
    enforce_gcp_family_policy: bool = True,
) -> WorkloadCUDResult:
    """
    Evaluate all commitment scenarios for a single resource workload.

    Parameters
    ----------
    workload_df : pd.DataFrame
        Filtered slice of usage data for a single resource_id.
    rates : Optional[Dict[str, float]]
        Custom discount rates dictionary.
    enforce_gcp_family_policy : bool
        Whether to check machine family policy for SUD on uncovered usage.

    Returns
    -------
    WorkloadCUDResult
    """
    if rates is None:
        rates = DEFAULT_CUD_RATES.copy()

    rate_1yr = rates.get("cud_1yr", DEFAULT_CUD_RATES["cud_1yr"])
    rate_3yr = rates.get("cud_3yr", DEFAULT_CUD_RATES["cud_3yr"])
    conserv_ratio = rates.get("conservative_coverage", DEFAULT_CUD_RATES["conservative_coverage"])

    res_id = str(workload_df["resource_id"].iloc[0])
    proj_id = str(workload_df["project_id"].iloc[0])
    mach_type = str(workload_df["machine_type"].iloc[0])
    family = extract_machine_family(mach_type)
    csv_sud_eligible = bool(workload_df["sud_eligible"].iloc[0])
    existing_commit_ratio = float(workload_df["existing_commitment"].iloc[0])

    # Verify SUD eligibility
    family_meta = GCP_SUD_FAMILY_POLICY.get(family)
    if enforce_gcp_family_policy and family_meta is not None:
        is_sud_eligible = family_meta["eligible"]
    else:
        is_sud_eligible = csv_sud_eligible

    # Historical metrics
    total_od_cost = float(workload_df["on_demand_cost"].sum())
    total_hours = float(workload_df["usage_hours"].sum())
    calculated_history = int(workload_df["calculated_history_days"].iloc[0]) if "calculated_history_days" in workload_df.columns else len(workload_df)
    period_days = max(calculated_history, 1)

    # Runtime fraction
    avg_daily_hours = total_hours / period_days
    run_fraction = min(1.0, avg_daily_hours / 24.0)

    # Existing commitment baseline: uses fixed historical contract rate (default 37%)
    existing_covered_od = total_od_cost * existing_commit_ratio
    existing_commit_rate = rates.get("existing_cud_rate", DEFAULT_CUD_RATES["cud_1yr"])
    existing_commit_cost = existing_covered_od * (1.0 - existing_commit_rate)
    uncovered_od_cost = total_od_cost * (1.0 - existing_commit_ratio)

    # Current state SUD on uncovered portion
    current_sud_rate = calculate_tiered_sud_discount_factor(run_fraction) if is_sud_eligible else 0.0
    current_sud_credit = round(uncovered_od_cost * current_sud_rate, 2)
    current_state_cost = round(existing_commit_cost + (uncovered_od_cost - current_sud_credit), 2)

    # Daily scaling factors for normalization
    daily_cost_factor = 1.0 / period_days
    monthly_factor = 30.0 / period_days
    annual_factor = 365.0 / period_days

    scenarios: Dict[str, ScenarioMetrics] = {}

    # Helper function to model a candidate commitment scenario on uncovered_od_cost
    def build_scenario(
        s_id: str,
        name: str,
        term: str,
        target_cov: float,
        discount_rate: float,
        assumption_text: str,
    ) -> ScenarioMetrics:
        # Portioned baseline to be newly committed
        newly_committed_od = uncovered_od_cost * target_cov
        new_commit_fee = newly_committed_od * (1.0 - discount_rate)

        # Remaining uncovered portion
        rem_uncovered_od = uncovered_od_cost * (1.0 - target_cov)
        # SUD applies strictly to the remaining uncovered portion
        rem_sud_credit = round(rem_uncovered_od * current_sud_rate, 2)
        rem_uncovered_net = rem_uncovered_od - rem_sud_credit

        # Total cost for this scenario = existing_commit + new_commit + remaining_uncovered_net
        total_scenario_cost = round(existing_commit_cost + new_commit_fee + rem_uncovered_net, 2)
        gross_savings_vs_od = round(total_od_cost - total_scenario_cost, 2)
        net_savings_vs_current = round(current_state_cost - total_scenario_cost, 2)

        # Break-even point for the newly committed block:
        # At what utilization fraction U does U * OnDemand == CommitFee?
        # U * OnDemand == OnDemand * (1 - discount_rate) => U = 1 - discount_rate
        break_even_pct = round((1.0 - discount_rate) * 100, 1) if discount_rate > 0 else 100.0

        return ScenarioMetrics(
            scenario_id=s_id,
            scenario_name=name,
            commitment_term=term,
            target_coverage_ratio=round(target_cov, 2),
            effective_discount_rate=round(discount_rate, 4),
            period_commitment_fee=round(existing_commit_cost + new_commit_fee, 2),
            period_uncovered_cost=round(rem_uncovered_od, 2),
            period_sud_credits_earned=round(rem_sud_credit, 2),
            period_total_cost=total_scenario_cost,
            period_gross_savings_vs_ondemand=gross_savings_vs_od,
            period_net_savings_vs_current_state=net_savings_vs_current,
            monthly_total_cost=round(total_scenario_cost * monthly_factor, 2),
            monthly_net_savings=round(net_savings_vs_current * monthly_factor, 2),
            annual_total_cost=round(total_scenario_cost * annual_factor, 2),
            annual_net_savings=round(net_savings_vs_current * annual_factor, 2),
            break_even_utilization_pct=break_even_pct,
            assumptions=assumption_text,
        )

    # 1. Baseline On-Demand (No discounts at all)
    scenarios["on_demand_baseline"] = ScenarioMetrics(
        scenario_id="on_demand_baseline",
        scenario_name="100% On-Demand Baseline",
        commitment_term="None",
        target_coverage_ratio=0.0,
        effective_discount_rate=0.0,
        period_commitment_fee=0.0,
        period_uncovered_cost=round(total_od_cost, 2),
        period_sud_credits_earned=0.0,
        period_total_cost=round(total_od_cost, 2),
        period_gross_savings_vs_ondemand=0.0,
        period_net_savings_vs_current_state=round(current_state_cost - total_od_cost, 2),
        monthly_total_cost=round(total_od_cost * monthly_factor, 2),
        monthly_net_savings=round((current_state_cost - total_od_cost) * monthly_factor, 2),
        annual_total_cost=round(total_od_cost * annual_factor, 2),
        annual_net_savings=round((current_state_cost - total_od_cost) * annual_factor, 2),
        break_even_utilization_pct=100.0,
        assumptions="Pure list price on-demand usage. No CUDs or SUDs applied.",
    )

    # 2. Current State (Status Quo)
    scenarios["current_state"] = ScenarioMetrics(
        scenario_id="current_state",
        scenario_name="Current State (Existing CUD + Active SUD)",
        commitment_term="Existing",
        target_coverage_ratio=0.0,
        effective_discount_rate=round(rate_1yr * existing_commit_ratio + current_sud_rate * (1 - existing_commit_ratio), 4),
        period_commitment_fee=round(existing_commit_cost, 2),
        period_uncovered_cost=round(uncovered_od_cost, 2),
        period_sud_credits_earned=current_sud_credit,
        period_total_cost=current_state_cost,
        period_gross_savings_vs_ondemand=round(total_od_cost - current_state_cost, 2),
        period_net_savings_vs_current_state=0.0,
        monthly_total_cost=round(current_state_cost * monthly_factor, 2),
        monthly_net_savings=0.0,
        annual_total_cost=round(current_state_cost * annual_factor, 2),
        annual_net_savings=0.0,
        break_even_utilization_pct=100.0,
        assumptions=f"Existing {int(existing_commit_ratio*100)}% commitment coverage + observed SUD credits on remainder.",
    )

    # 3. 1-Year CUD Full Coverage (100% of remaining uncovered usage)
    scenarios["cud_1yr_full"] = build_scenario(
        s_id="cud_1yr_full",
        name="1-Year CUD (100% Uncovered Base)",
        term="1-Year",
        target_cov=1.0,
        discount_rate=rate_1yr,
        assumption_text=f"Fixed 1-Year commitment on 100% of remaining usage at illustrative {int(rate_1yr*100)}% discount. Replaces SUD.",
    )

    # 4. 3-Year CUD Full Coverage (100% of remaining uncovered usage)
    scenarios["cud_3yr_full"] = build_scenario(
        s_id="cud_3yr_full",
        name="3-Year CUD (100% Uncovered Base)",
        term="3-Year",
        target_cov=1.0,
        discount_rate=rate_3yr,
        assumption_text=f"Fixed 3-Year commitment on 100% of remaining usage at illustrative {int(rate_3yr*100)}% discount. Replaces SUD.",
    )

    # 5. Conservative CUD Scenario (e.g. 70% coverage at 1-Year rate)
    scenarios["cud_conservative"] = build_scenario(
        s_id="cud_conservative",
        name=f"Conservative CUD ({int(conserv_ratio*100)}% Coverage, 1-Year)",
        term="1-Year",
        target_cov=conserv_ratio,
        discount_rate=rate_1yr,
        assumption_text=f"Commit to {int(conserv_ratio*100)}% of baseline at {int(rate_1yr*100)}% discount. Leaves {int((1-conserv_ratio)*100)}% flexible buffer.",
    )

    # 6. Flexible Strategy (No new commitment)
    scenarios["flexible_no_commit"] = ScenarioMetrics(
        scenario_id="flexible_no_commit",
        scenario_name="Maintain Flexibility (No New Commit)",
        commitment_term="None",
        target_coverage_ratio=0.0,
        effective_discount_rate=current_sud_rate,
        period_commitment_fee=round(existing_commit_cost, 2),
        period_uncovered_cost=round(uncovered_od_cost, 2),
        period_sud_credits_earned=current_sud_credit,
        period_total_cost=current_state_cost,
        period_gross_savings_vs_ondemand=round(total_od_cost - current_state_cost, 2),
        period_net_savings_vs_current_state=0.0,
        monthly_total_cost=round(current_state_cost * monthly_factor, 2),
        monthly_net_savings=0.0,
        annual_total_cost=round(current_state_cost * annual_factor, 2),
        annual_net_savings=0.0,
        break_even_utilization_pct=0.0,
        assumptions="0% commitment liability. Maximize operational agility and retain SUD credits.",
    )

    # Determine highest savings scenario
    # Default to flexible unless a commitment provides positive net incremental savings
    best_id = "flexible_no_commit"
    max_net = 0.0
    for cand_id in ["cud_conservative", "cud_1yr_full", "cud_3yr_full"]:
        cand_net = scenarios[cand_id].period_net_savings_vs_current_state
        if cand_net > max_net:
            max_net = cand_net
            best_id = cand_id

    return WorkloadCUDResult(
        resource_id=res_id,
        project_id=proj_id,
        machine_type=mach_type,
        machine_family=family,
        is_sud_eligible=is_sud_eligible,
        history_days=calculated_history,
        period_days=period_days,
        existing_commitment_ratio=round(existing_commit_ratio, 2),
        total_on_demand_baseline=round(total_od_cost, 2),
        existing_commitment_cost=round(existing_commit_cost, 2),
        uncovered_on_demand_cost=round(uncovered_od_cost, 2),
        current_state_cost=current_state_cost,
        current_state_sud_credit=current_sud_credit,
        scenarios=scenarios,
        best_financial_scenario=best_id,
    )


def analyze_cud_portfolio(
    df: pd.DataFrame,
    rates: Optional[Dict[str, float]] = None,
    enforce_gcp_family_policy: bool = True,
) -> PortfolioCUDReport:
    """
    Run full CUD scenario comparisons across all workloads in the dataset.

    Returns
    -------
    PortfolioCUDReport
        Aggregated report with workload details and total portfolio financial comparisons.
    """
    if rates is None:
        rates = DEFAULT_CUD_RATES.copy()

    workload_results: List[WorkloadCUDResult] = []
    portfolio_scenarios: Dict[str, Dict[str, float]] = {
        "on_demand_baseline": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
        "current_state": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
        "cud_1yr_full": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
        "cud_3yr_full": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
        "cud_conservative": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
        "flexible_no_commit": {"period_cost": 0.0, "monthly_cost": 0.0, "net_savings": 0.0},
    }

    for res_id, group in df.groupby("resource_id"):
        res = evaluate_workload_cud_scenarios(
            workload_df=group,
            rates=rates,
            enforce_gcp_family_policy=enforce_gcp_family_policy,
        )
        workload_results.append(res)

        for s_key, s_data in res.scenarios.items():
            if s_key in portfolio_scenarios:
                portfolio_scenarios[s_key]["period_cost"] += s_data.period_total_cost
                portfolio_scenarios[s_key]["monthly_cost"] += s_data.monthly_total_cost
                portfolio_scenarios[s_key]["net_savings"] += s_data.monthly_net_savings

    # Round aggregated values
    for s_key in portfolio_scenarios:
        portfolio_scenarios[s_key]["period_cost"] = round(portfolio_scenarios[s_key]["period_cost"], 2)
        portfolio_scenarios[s_key]["monthly_cost"] = round(portfolio_scenarios[s_key]["monthly_cost"], 2)
        portfolio_scenarios[s_key]["net_savings"] = round(portfolio_scenarios[s_key]["net_savings"], 2)

    notes = [
        "Illustrative CUD discount rates applied: 1-Year = 37%, 3-Year = 55%, Conservative Coverage = 70%.",
        "Fixed-price scenario formula applied: Net Savings = Covered On-Demand Equivalent (A) - Commitment Fee (C) - Opportunity Cost of Displaced SUD.",
        "Existing commitments take priority; new CUDs only evaluated on remaining uncovered usage.",
        "Uncovered usage in partial commitment scenarios continues to receive SUD if eligible and run-fraction permits.",
    ]

    return PortfolioCUDReport(
        workload_results=workload_results,
        portfolio_scenarios=portfolio_scenarios,
        configured_rates=rates,
        methodology_notes=notes,
    )
