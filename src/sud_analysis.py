"""
Sustained Use Discount (SUD) Analysis Module for CloudWise.

Analyzes Compute Engine usage for Sustained Use Discounts (SUD).
Enforces realistic Google Cloud FinOps rules:
- Tiered discount structure (0-25%: 0%, 25-50%: 20%, 50-75%: 40%, 75-100%: 60%).
- Machine family eligibility verification (e.g., E2, C3, and Tau families do not receive SUD).
- Excludes hours covered by existing CUDs to prevent discount double-counting.
- Distinguishes observed actual SUD savings from theoretical full-month maximums.
- Documents exact Cloud Billing BigQuery schema fields required for real billing reconciliation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Known GCP Machine Family SUD Eligibility Matrix
# Reference: Google Cloud Compute Engine Sustained Use Discounts documentation
GCP_SUD_FAMILY_POLICY = {
    # Eligible families (general-purpose v1/v2, compute-optimized, memory-optimized)
    "n1": {"eligible": True, "max_blended_discount": 0.30, "family_name": "N1 (Predefined & Custom)"},
    "n2": {"eligible": True, "max_blended_discount": 0.30, "family_name": "N2 Intel Xeon"},
    "n2d": {"eligible": True, "max_blended_discount": 0.30, "family_name": "N2D AMD EPYC"},
    "c2": {"eligible": True, "max_blended_discount": 0.30, "family_name": "C2 Compute-Optimized"},
    "c2d": {"eligible": True, "max_blended_discount": 0.30, "family_name": "C2D Compute-Optimized AMD"},
    "m1": {"eligible": True, "max_blended_discount": 0.30, "family_name": "M1 Memory-Optimized"},
    "m2": {"eligible": True, "max_blended_discount": 0.30, "family_name": "M2 Memory-Optimized"},
    # Ineligible families (cost-optimized, next-gen architectures, burstable)
    "e2": {"eligible": False, "max_blended_discount": 0.0, "family_name": "E2 Cost-Optimized (No SUDs in GCP)"},
    "c3": {"eligible": False, "max_blended_discount": 0.0, "family_name": "C3 Intel Sapphire Rapids (CUD only)"},
    "c3d": {"eligible": False, "max_blended_discount": 0.0, "family_name": "C3D AMD Genoa (CUD only)"},
    "t2d": {"eligible": False, "max_blended_discount": 0.0, "family_name": "Tau T2D (CUD only)"},
    "t2a": {"eligible": False, "max_blended_discount": 0.0, "family_name": "Tau T2A Arm (CUD only)"},
    "g2": {"eligible": False, "max_blended_discount": 0.0, "family_name": "G2 GPU instances (GPU/CUD only)"},
    "a2": {"eligible": False, "max_blended_discount": 0.0, "family_name": "A2 GPU instances (GPU/CUD only)"},
}


@dataclass
class WorkloadSUDResult:
    """Detailed SUD evaluation for an individual workload."""
    resource_id: str
    project_id: str
    machine_type: str
    machine_family: str
    is_policy_eligible: bool
    csv_stated_eligible: bool
    eligibility_note: str
    total_usage_hours: float
    total_on_demand_cost: float
    effective_monthly_run_ratio: float  # runtime % of 730h month
    uncovered_usage_ratio: float  # 1.0 - existing_commitment
    observed_sud_discount_rate: float
    observed_sud_savings: float
    net_cost_after_sud: float
    theoretical_max_sud_savings: float
    unrealized_sud_gap: float


@dataclass
class SUDAnalysisReport:
    """Aggregate SUD evaluation report across all workloads."""
    workload_results: List[WorkloadSUDResult]
    summary_metrics: Dict[str, Any]
    real_billing_requirements: List[str]
    policy_overrides: List[str] = field(default_factory=list)


def extract_machine_family(machine_type: str) -> str:
    """Extract machine family prefix from machine type string (e.g. 'n2-standard-8' -> 'n2')."""
    if not machine_type or not isinstance(machine_type, str):
        return "unknown"
    clean = machine_type.strip().lower()
    prefix = clean.split("-")[0]
    return prefix


def calculate_tiered_sud_discount_factor(monthly_run_fraction: float) -> float:
    """
    Calculate Google Cloud standard tiered SUD effective discount fraction
    for compute instances running a given fraction of a standard month (0.0 to 1.0).

    Tiers:
    - 0% - 25% of month (0 to 0.25): 0% discount
    - 25% - 50% of month (0.25 to 0.50): 20% discount on incremental hours
    - 50% - 75% of month (0.50 to 0.75): 40% discount on incremental hours
    - 75% - 100% of month (0.75 to 1.00): 60% discount on incremental hours

    Returns effective blended discount fraction across all hours run (0.0 to 0.30).
    """
    f = float(np.clip(monthly_run_fraction, 0.0, 1.0))
    if f <= 0.0:
        return 0.0

    # Total cost factor = sum(hours_in_tier * (1 - tier_discount)) / total_hours
    tier1_hours = min(f, 0.25)
    tier2_hours = max(0.0, min(f - 0.25, 0.25))
    tier3_hours = max(0.0, min(f - 0.50, 0.25))
    tier4_hours = max(0.0, min(f - 0.75, 0.25))

    weighted_cost_ratio = (
        (tier1_hours * 1.00) +
        (tier2_hours * 0.80) +
        (tier3_hours * 0.60) +
        (tier4_hours * 0.40)
    ) / f

    effective_discount = 1.0 - weighted_cost_ratio
    return round(float(effective_discount), 4)


def analyze_sud(
    df: pd.DataFrame,
    enforce_gcp_family_policy: bool = True,
    standard_month_hours: float = 730.0,
) -> SUDAnalysisReport:
    """
    Perform comprehensive Sustained Use Discount analysis across all workloads in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Validated compute usage dataset from load_usage_data.
    enforce_gcp_family_policy : bool
        If True, validates machine_type against real Google Cloud family eligibility
        instead of blindly trusting the dataset's sud_eligible flag.
    standard_month_hours : float
        Average monthly hours for billing normalization (default 730.0).

    Returns
    -------
    SUDAnalysisReport
        Structured report containing per-workload and portfolio-level SUD calculations.
    """
    workload_results: List[WorkloadSUDResult] = []
    policy_overrides: List[str] = []

    total_baseline_cost = 0.0
    total_eligible_cost = 0.0
    total_observed_sud_savings = 0.0
    total_theoretical_max_sud = 0.0

    for res_id, group in df.groupby("resource_id"):
        proj_id = str(group["project_id"].iloc[0])
        mach_type = str(group["machine_type"].iloc[0])
        family = extract_machine_family(mach_type)
        csv_stated_eligible = bool(group["sud_eligible"].iloc[0])
        existing_commit_ratio = float(group["existing_commitment"].iloc[0])

        total_hours = float(group["usage_hours"].sum())
        total_cost = float(group["on_demand_cost"].sum())
        history_days = float(group["calculated_history_days"].iloc[0]) if "calculated_history_days" in group.columns else 90.0

        # Determine real-world policy eligibility
        family_meta = GCP_SUD_FAMILY_POLICY.get(family, None)
        if family_meta is not None:
            policy_eligible = family_meta["eligible"]
        else:
            # Unknown custom family: fall back to CSV flag but document caution
            policy_eligible = csv_stated_eligible

        eligibility_note = ""
        effective_eligible = csv_stated_eligible
        if enforce_gcp_family_policy and family_meta is not None:
            if csv_stated_eligible and not family_meta["eligible"]:
                policy_overrides.append(
                    f"Workload '{res_id}' ({mach_type}): CSV flagged sud_eligible=True, "
                    f"but GCP {family.upper()} family is strictly ineligible for SUDs. Overridden to Ineligible."
                )
                effective_eligible = False
                eligibility_note = f"Ineligible (GCP {family.upper()} does not receive SUD)"
            elif not csv_stated_eligible and family_meta["eligible"]:
                eligibility_note = f"Policy-eligible {family.upper()}, but flagged ineligible in dataset"
                effective_eligible = False
            elif effective_eligible:
                eligibility_note = f"Eligible {family_meta['family_name']}"
            else:
                eligibility_note = f"Ineligible {family_meta['family_name']}"
        else:
            effective_eligible = csv_stated_eligible
            eligibility_note = "Eligible per dataset" if effective_eligible else "Ineligible per dataset"

        # Calculate average monthly runtime fraction
        # Average hours per day / 24.0
        avg_daily_hours = total_hours / max(history_days, 1.0)
        run_fraction = min(1.0, avg_daily_hours / 24.0)

        # Uncovered usage portion: CUD takes precedence over SUD
        # SUD can only apply to hours not already covered by CUD commitments
        uncovered_ratio = max(0.0, 1.0 - existing_commit_ratio)

        if effective_eligible and uncovered_ratio > 0.0:
            effective_discount_rate = calculate_tiered_sud_discount_factor(run_fraction)
            # Savings applied strictly to uncovered portion
            observed_savings = round(total_cost * uncovered_ratio * effective_discount_rate, 2)
            # Theoretical max if instance ran 100% of the time (30% discount on uncovered portion)
            max_theoretical_rate = 0.30
            theoretical_max_savings = round(total_cost * uncovered_ratio * max_theoretical_rate, 2)
            unrealized_gap = max(0.0, round(theoretical_max_savings - observed_savings, 2))
        else:
            effective_discount_rate = 0.0
            observed_savings = 0.0
            theoretical_max_savings = 0.0
            unrealized_gap = 0.0

        net_cost = round(total_cost - observed_savings, 2)

        total_baseline_cost += total_cost
        if effective_eligible:
            total_eligible_cost += (total_cost * uncovered_ratio)
        total_observed_sud_savings += observed_savings
        total_theoretical_max_sud += theoretical_max_savings

        workload_results.append(
            WorkloadSUDResult(
                resource_id=res_id,
                project_id=proj_id,
                machine_type=mach_type,
                machine_family=family,
                is_policy_eligible=effective_eligible,
                csv_stated_eligible=csv_stated_eligible,
                eligibility_note=eligibility_note,
                total_usage_hours=round(total_hours, 1),
                total_on_demand_cost=round(total_cost, 2),
                effective_monthly_run_ratio=round(run_fraction, 3),
                uncovered_usage_ratio=round(uncovered_ratio, 2),
                observed_sud_discount_rate=round(effective_discount_rate, 4),
                observed_sud_savings=observed_savings,
                net_cost_after_sud=net_cost,
                theoretical_max_sud_savings=theoretical_max_savings,
                unrealized_sud_gap=unrealized_gap,
            )
        )

    summary_metrics = {
        "total_baseline_on_demand_cost": round(total_baseline_cost, 2),
        "total_eligible_uncovered_cost": round(total_eligible_cost, 2),
        "total_observed_sud_savings": round(total_observed_sud_savings, 2),
        "total_theoretical_max_sud": round(total_theoretical_max_sud, 2),
        "portfolio_effective_sud_discount_pct": round(
            (total_observed_sud_savings / max(total_baseline_cost, 1.0)) * 100, 2
        ),
        "eligible_workloads_count": sum(1 for w in workload_results if w.is_policy_eligible),
        "ineligible_workloads_count": sum(1 for w in workload_results if not w.is_policy_eligible),
    }

    real_billing_requirements = [
        "Google Cloud Billing Export to BigQuery dataset (standard or detailed export).",
        "Field `credits` array filtered by `credits.type = 'SUSTAINED_USAGE_DISCOUNT'`.",
        "Field `sku.description` and `sku.id` to separate vCPU, memory, and local SSD usage.",
        "Field `usage.amount_in_pricing_units` and `usage.pricing_unit` (e.g. core-hours, byte-seconds).",
        "Field `labels` or `system_labels` to track machine family and commitment association.",
        "Billing account currency and effective list pricing table (`cloud_billing_pricing_export`).",
    ]

    return SUDAnalysisReport(
        workload_results=workload_results,
        summary_metrics=summary_metrics,
        real_billing_requirements=real_billing_requirements,
        policy_overrides=policy_overrides,
    )
