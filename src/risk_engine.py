"""
Risk, Stress-Testing, and Confidence Engine for CloudWise.

Evaluates financial risk and confidence for cloud commitments:
1. Simulates future usage volatility across stress levels: 60%, 75%, 90%, 100%, 120%.
2. Computes unused commitment exposure (shelfware penalty), utilization %, and net ROI.
3. Provides transparent, explainable confidence assessments based on history length,
   coefficient of variation (CV), trend direction, and data completeness.
4. Identifies situations where the correct FinOps recommendation is to WAIT for more data
   or avoid commitments altogether (e.g. declining workloads, bursty traffic, dev clusters).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .sud_analysis import calculate_tiered_sud_discount_factor, GCP_SUD_FAMILY_POLICY, extract_machine_family


STRESS_MULTIPLIERS = [0.60, 0.75, 0.90, 1.00, 1.20]


@dataclass
class StressScenarioResult:
    """Outcome of stress-testing a commitment at a specific future usage level."""
    multiplier: float  # e.g. 0.60, 0.75, etc.
    label: str  # "-40% Decline", "Expected (100%)", etc.
    projected_usage_hours: float
    projected_on_demand_baseline_cost: float
    scenario_total_cost: float
    commitment_utilization_pct: float
    unused_commitment_exposure: float  # financial waste from unutilized commitment
    net_savings_vs_baseline: float
    is_financially_beneficial: bool
    explanation: str


@dataclass
class ConfidenceAssessment:
    """Transparent, explainable confidence scoring breakdown."""
    overall_score: int  # 0 to 100
    confidence_level: str  # "High", "Medium", "Low", "Insufficient Evidence"
    history_days: int
    history_score: int  # max 30
    cv_score: int  # max 30
    trend_score: int  # max 15
    completeness_score: int  # max 15
    existing_coverage_score: int  # max 10
    coefficient_of_variation: float
    trend_direction: str  # "Stable", "Declining", "Expanding", "Erratic"
    data_completeness_pct: float
    rationale_bullets: List[str]


@dataclass
class WorkloadRecommendation:
    """Final synthesis recommendation for a workload."""
    resource_id: str
    project_id: str
    machine_type: str
    recommended_action: str  # "1-Year CUD", "3-Year CUD", "Conservative CUD", "Maintain Flexibility / SUD", "Wait (Insufficient Data)"
    action_type: str  # "COMMIT", "FLEXIBLE", "WAIT", "NO_CHANGE"
    confidence: ConfidenceAssessment
    break_even_utilization_pct: float
    estimated_monthly_savings: float
    estimated_annual_savings: float
    stress_test_results: List[StressScenarioResult]
    primary_reason: str
    downside_risk_warning: str
    assumptions_and_limitations: List[str]


def compute_usage_volatility(hours_series: pd.Series) -> Tuple[float, float, float]:
    """Compute mean, standard deviation, and Coefficient of Variation (CV)."""
    clean_series = hours_series.dropna()
    if len(clean_series) < 2:
        return 0.0, 0.0, 0.0
    mean_val = float(clean_series.mean())
    std_val = float(clean_series.std(ddof=1))
    if mean_val <= 0.001:
        cv = 0.0
    else:
        cv = std_val / mean_val
    return round(mean_val, 2), round(std_val, 2), round(cv, 4)


def detect_usage_trend(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Detect whether usage is stable, declining, expanding, or erratic by comparing
    early period usage vs recent period usage.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    if n < 14:
        return "Insufficient History", 0.0

    split = max(7, n // 3)
    early_mean = float(df_sorted.head(split)["usage_hours"].mean())
    late_mean = float(df_sorted.tail(split)["usage_hours"].mean())

    if early_mean <= 0.01:
        return "Expanding", 1.0

    change_ratio = (late_mean - early_mean) / early_mean

    if change_ratio <= -0.25:
        return "Declining", round(change_ratio, 2)
    elif change_ratio >= 0.25:
        return "Expanding", round(change_ratio, 2)
    elif abs(change_ratio) < 0.15:
        return "Stable", round(change_ratio, 2)
    else:
        return "Moderate Shift", round(change_ratio, 2)


def evaluate_confidence(
    df: pd.DataFrame,
    history_days: int,
    completeness_pct: float,
    existing_commit_ratio: float,
) -> ConfidenceAssessment:
    """
    Produce a transparent, rule-based confidence score (0-100) and tier.
    Score is not a black-box probability; it maps to explicit operational telemetry checks.
    """
    mean_h, std_h, cv = compute_usage_volatility(df["usage_hours"])
    trend_label, trend_change = detect_usage_trend(df)
    rationale: List[str] = []

    # 1. History Length (Max 30)
    if history_days < 30:
        history_score = 5
        rationale.append(f"Insufficient history: {history_days} calendar days is below the 30-day minimum observation threshold.")
    elif history_days < 60:
        history_score = 18
        rationale.append(f"Moderate history: {history_days} days observed; provides preliminary baseline but misses quarterly seasonality.")
    else:
        history_score = 30
        rationale.append(f"Strong history: {history_days} days observed; robust basis for evaluating steady-state commitments.")

    # 2. Usage Volatility / CV (Max 30)
    if cv < 0.10:
        cv_score = 30
        rationale.append(f"Ultra-low volatility (CV={cv:.2f}): instance runs flat 24/7, optimal for commitment lock-in.")
    elif cv < 0.25:
        cv_score = 22
        rationale.append(f"Low-to-moderate volatility (CV={cv:.2f}): minor usage swings; suitable for conservative commitment.")
    elif cv < 0.45:
        cv_score = 12
        rationale.append(f"Moderate volatility (CV={cv:.2f}): notable variability; dev/test or cyclical patterns present.")
    else:
        cv_score = 4
        rationale.append(f"High volatility (CV={cv:.2f}): erratic or bursty spikes; commitment risks shelfware during quiet periods.")

    # 3. Trend Direction (Max 15)
    if trend_label == "Declining":
        trend_score = 2
        rationale.append(f"Declining trend ({int(trend_change*100)}% shift): workload is shrinking or decommissioning; high risk of shelfware!")
    elif trend_label == "Stable":
        trend_score = 15
        rationale.append("Stable trend: demand is steady over time with no decommissioning indicators.")
    elif trend_label == "Expanding":
        trend_score = 12
        rationale.append("Expanding trend: demand is growing; commitment base is safe but size conservatively.")
    else:
        trend_score = 8
        rationale.append(f"Variable trend ({trend_label}): mild demand drift observed.")

    # 4. Data Completeness (Max 15)
    if completeness_pct >= 95.0:
        comp_score = 15
        rationale.append(f"High data completeness: {completeness_pct}% of calendar days recorded without telemetry gaps.")
    elif completeness_pct >= 80.0:
        comp_score = 10
        rationale.append(f"Fair data completeness: {completeness_pct}% recorded; some missing observation days.")
    else:
        comp_score = 4
        rationale.append(f"Telemetry gaps: only {completeness_pct}% days recorded; risk of unmonitored downtime.")

    # 5. Existing Coverage Headroom (Max 10)
    if existing_commit_ratio >= 0.85:
        exist_score = 3
        rationale.append(f"Existing commitment saturated: already {int(existing_commit_ratio*100)}% committed; little incremental headroom.")
    elif existing_commit_ratio >= 0.50:
        exist_score = 7
        rationale.append(f"Partial existing coverage: {int(existing_commit_ratio*100)}% already committed; incremental sizing required.")
    else:
        exist_score = 10
        rationale.append("Uncommitted baseline: full headroom available for evaluation.")

    total_score = history_score + cv_score + trend_score + comp_score + exist_score

    # Determine confidence level
    # Hard cap: if history is < 30 days, confidence CANNOT be High or Medium
    if history_days < 30:
        confidence_level = "Insufficient Evidence"
    elif total_score >= 75:
        confidence_level = "High"
    elif total_score >= 50:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    return ConfidenceAssessment(
        overall_score=total_score,
        confidence_level=confidence_level,
        history_days=history_days,
        history_score=history_score,
        cv_score=cv_score,
        trend_score=trend_score,
        completeness_score=comp_score,
        existing_coverage_score=exist_score,
        coefficient_of_variation=cv,
        trend_direction=trend_label,
        data_completeness_pct=completeness_pct,
        rationale_bullets=rationale,
    )


def simulate_commitment_stress(
    baseline_uncovered_od: float,
    target_coverage_ratio: float,
    discount_rate: float,
    is_sud_eligible: bool,
    avg_daily_hours: float,
    existing_commit_fee: float,
    period_days: int,
) -> List[StressScenarioResult]:
    """
    Stress-test a proposed commitment scenario under future usage shifts:
    60%, 75%, 90%, 100%, 120%.

    Demonstrates downside financial risk and unused commitment exposure.
    """
    results: List[StressScenarioResult] = []

    # Newly committed on-demand chunk
    target_commit_chunk = baseline_uncovered_od * target_coverage_ratio
    new_commit_fee = target_commit_chunk * (1.0 - discount_rate)
    total_commit_fee = existing_commit_fee + new_commit_fee

    for mult in STRESS_MULTIPLIERS:
        pct_label = f"{int(mult * 100)}%"
        if mult == 1.00:
            label = "100% (Expected Base)"
        elif mult < 1.00:
            label = f"{pct_label} ({int((mult-1.0)*100)}% Drop)"
        else:
            label = f"{pct_label} (+{int((mult-1.0)*100)}% Surge)"

        # Future uncovered demand at this multiplier
        future_demand_od = baseline_uncovered_od * mult

        # How much of future demand is absorbed by the new commitment chunk?
        if target_commit_chunk > 0:
            if future_demand_od >= target_commit_chunk:
                # 100% utilized, remaining spills over to on-demand
                utilization_pct = 100.0
                unused_exposure = 0.0
                spillover_od = future_demand_od - target_commit_chunk
            else:
                # Underutilized: demand is less than commitment size!
                utilization_pct = round((future_demand_od / target_commit_chunk) * 100, 1)
                # Unused commitment dollars
                unused_exposure = round((1.0 - (utilization_pct / 100.0)) * new_commit_fee, 2)
                spillover_od = 0.0
        else:
            utilization_pct = 0.0
            unused_exposure = 0.0
            spillover_od = future_demand_od

        # SUD on spillover (if eligible and running enough hours)
        run_ratio = min(1.0, (avg_daily_hours * mult) / 24.0)
        sud_factor = calculate_tiered_sud_discount_factor(run_ratio) if is_sud_eligible else 0.0
        spillover_sud_credit = round(spillover_od * sud_factor, 2)
        spillover_net = spillover_od - spillover_sud_credit

        # Total cost for this scenario under multiplier
        scenario_cost = round(total_commit_fee + spillover_net, 2)

        # Baseline alternative without new commitment at same multiplier
        # (existing commitment + future demand at on-demand minus SUD)
        base_demand_sud = round(future_demand_od * sud_factor, 2) if is_sud_eligible else 0.0
        projected_base_cost = round(existing_commit_fee + (future_demand_od - base_demand_sud), 2)

        net_savings = round(projected_base_cost - scenario_cost, 2)
        is_beneficial = net_savings > 0

        if target_commit_chunk <= 0:
            explanation = "No incremental commitment active (0% coverage). 100% flexible on-demand / SUD."
        elif unused_exposure > 0:
            explanation = (
                f"Underutilized at {utilization_pct}% capacity. Wasting ${unused_exposure:.2f} in shelfware. "
                f"Net financial outcome: {'+$' if net_savings >= 0 else '-$'}{abs(net_savings):.2f} vs flexible alternative."
            )
        elif spillover_od > 0:
            explanation = (
                f"100% commitment capacity utilized. ${spillover_od:.2f} spilled over to elastic on-demand. "
                f"Net savings: +${net_savings:.2f}."
            )
        else:
            explanation = f"100% utilized with 0 spillover. Net savings: +${net_savings:.2f}."

        results.append(
            StressScenarioResult(
                multiplier=mult,
                label=label,
                projected_usage_hours=round(avg_daily_hours * period_days * mult, 1),
                projected_on_demand_baseline_cost=projected_base_cost,
                scenario_total_cost=scenario_cost,
                commitment_utilization_pct=utilization_pct,
                unused_commitment_exposure=unused_exposure,
                net_savings_vs_baseline=net_savings,
                is_financially_beneficial=is_beneficial,
                explanation=explanation,
            )
        )

    return results


def generate_recommendation_for_workload(
    workload_df: pd.DataFrame,
    rates: Optional[Dict[str, float]] = None,
    enforce_gcp_family_policy: bool = True,
) -> WorkloadRecommendation:
    """
    Generate an explainable FinOps recommendation for a workload, integrating
    confidence assessment, risk simulations, and FinOps constraints.
    """
    from .cud_analysis import evaluate_workload_cud_scenarios, DEFAULT_CUD_RATES
    if rates is None:
        rates = DEFAULT_CUD_RATES.copy()

    cud_res = evaluate_workload_cud_scenarios(
        workload_df=workload_df,
        rates=rates,
        enforce_gcp_family_policy=enforce_gcp_family_policy,
    )

    history_days = cud_res.history_days
    completeness_pct = float(workload_df["data_completeness_pct"].iloc[0]) if "data_completeness_pct" in workload_df.columns else 100.0
    existing_cov = cud_res.existing_commitment_ratio

    confidence = evaluate_confidence(
        df=workload_df,
        history_days=history_days,
        completeness_pct=completeness_pct,
        existing_commit_ratio=existing_cov,
    )

    avg_hours = float(workload_df["usage_hours"].mean())
    cv = confidence.coefficient_of_variation
    trend = confidence.trend_direction

    # Scenario financial metrics
    c_1yr = cud_res.scenarios["cud_1yr_full"]
    c_3yr = cud_res.scenarios["cud_3yr_full"]
    c_conserv = cud_res.scenarios["cud_conservative"]

    rate_1yr = rates.get("cud_1yr", DEFAULT_CUD_RATES["cud_1yr"])
    rate_3yr = rates.get("cud_3yr", DEFAULT_CUD_RATES["cud_3yr"])
    conserv_ratio = rates.get("conservative_coverage", DEFAULT_CUD_RATES["conservative_coverage"])

    # Decision logic
    action = "Maintain Flexibility / SUD"
    action_type = "FLEXIBLE"
    rec_target_cov = 0.0
    rec_rate = 0.0
    primary_reason = ""
    downside_warning = ""
    est_monthly_savings = 0.0
    est_annual_savings = 0.0
    break_even = 100.0

    # Rule 1: Short history -> MUST RECOMMEND WAIT
    if history_days < 30:
        action = "Wait (Insufficient History)"
        action_type = "WAIT"
        primary_reason = (
            f"Observed history is only {history_days} calendar days, well below the required 30-day FinOps threshold. "
            "Entering a 1-year or 3-year legally binding commitment with under 30 days of telemetry carries high risk of mis-sizing. "
            "Recommendation: Maintain On-Demand / SUD and accumulate at least 60 days of data."
        )
        downside_warning = "High risk of premature commitment lock-in on unvalidated workload trajectory."
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        break_even = round((1.0 - rate_1yr) * 100, 1)

    # Rule 2: Declining trend -> Do Not Commit
    elif trend == "Declining":
        action = "Do Not Commit (Declining Demand)"
        action_type = "FLEXIBLE"
        primary_reason = (
            f"Workload usage is sharply declining (downward trend). Purchasing a multi-year commitment now "
            "would create severe shelfware waste as the workload shrinks further. "
            "Recommendation: Rely on automatic SUDs and decommission instances as demand drops."
        )
        downside_warning = "Extreme downside: 100% of unused commitment fee becomes pure unrecoverable financial loss."
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        break_even = round((1.0 - rate_1yr) * 100, 1)

    # Rule 3: High existing coverage -> Already Sufficient
    elif existing_cov >= 0.85:
        action = "Maintain Existing Commitment"
        action_type = "NO_CHANGE"
        primary_reason = (
            f"Workload already possesses {int(existing_cov*100)}% commitment coverage. Remaining headroom is too small "
            "to warrant an additional commitment tranche. Maintain current coverage."
        )
        downside_warning = "Over-committing beyond baseline leaves zero elasticity for maintenance or down-scaling."
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        break_even = 100.0

    # Rule 4: Bursty / Spiky usage (CV > 0.40) -> Flexible
    elif cv > 0.40:
        action = "Maintain Flexibility (SUD / On-Demand)"
        action_type = "FLEXIBLE"
        primary_reason = (
            f"High usage volatility (CV={cv:.2f}). Workload exhibits irregular spikes rather than steady baseline compute. "
            "Average runtime is well below the commitment break-even utilization threshold. "
            "Recommendation: Keep flexible on-demand or explore Spot VMs for bursty processing."
        )
        downside_warning = f"A 1-Year CUD requires at least {int((1-rate_1yr)*100)}% steady utilization to break even; spiky workloads fail this test."
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        break_even = round((1.0 - rate_1yr) * 100, 1)

    # Rule 5: Dev / Test business hours (workloads running only business hours, <12 hrs/day average)
    elif avg_hours < 12.0:
        action = "Schedule VM Shutdowns (Do Not Commit)"
        action_type = "FLEXIBLE"
        primary_reason = (
            f"Workload operates on a business-hours schedule (~{avg_hours:.1f} hrs/day average). "
            "FinOps Best Practice: Implementing Cloud Scheduler / Cloud Functions to shut down VMs on nights and weekends "
            "saves ~65% on on-demand cost, significantly outperforming a 37% CUD without entering a 24/7 commitment contract."
        )
        downside_warning = "Entering a 24/7 CUD on a part-time dev workload creates ~50%+ wasted capacity during non-business hours."
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        break_even = round((1.0 - rate_1yr) * 100, 1)

    # Rule 6: Stable Production Workload with moderate-to-high confidence
    elif cv < 0.15 and confidence.confidence_level in ["High", "Medium"] and trend == "Stable":
        # Check if 3-Year CUD produces strictly positive net savings and is superior
        if (
            c_3yr.monthly_net_savings > 0.0
            and rate_3yr > 0.0
            and c_3yr.monthly_net_savings > c_1yr.monthly_net_savings
            and c_3yr.monthly_net_savings >= c_conserv.monthly_net_savings
            and existing_cov == 0.0
        ):
            action = "3-Year CUD (High Confidence)"
            action_type = "COMMIT"
            rec_target_cov = 1.0
            rec_rate = rate_3yr
            primary_reason = (
                f"Workload is exceptionally stable (CV={cv:.2f}, avg {avg_hours:.1f} hrs/day) with {history_days} days of clean telemetry. "
                f"A 3-Year CUD yields maximum net savings of ${c_3yr.monthly_net_savings:.2f}/mo ({int(rate_3yr*100)}% illustrative discount), "
                "surpassing SUD and 1-Year alternatives."
            )
            downside_warning = f"Commitment break-even is {c_3yr.break_even_utilization_pct}%. Requires multi-year architectural stability."
            est_monthly_savings = c_3yr.monthly_net_savings
            est_annual_savings = c_3yr.annual_net_savings
            break_even = c_3yr.break_even_utilization_pct

        elif c_conserv.monthly_net_savings > 0.0 and rate_1yr > 0.0 and conserv_ratio > 0.0:
            action = f"Conservative 1-Year CUD ({int(conserv_ratio*100)}% Coverage)"
            action_type = "COMMIT"
            rec_target_cov = conserv_ratio
            rec_rate = rate_1yr
            primary_reason = (
                f"Stable workload with existing partial coverage or moderate term horizon. "
                f"Committing {int(conserv_ratio*100)}% of remaining baseline secures ${c_conserv.monthly_net_savings:.2f}/mo net savings "
                f"while retaining a {int((1-conserv_ratio)*100)}% elastic buffer for maintenance."
            )
            downside_warning = f"Break-even utilization is {c_conserv.break_even_utilization_pct}% of the committed tier."
            est_monthly_savings = c_conserv.monthly_net_savings
            est_annual_savings = c_conserv.annual_net_savings
            break_even = c_conserv.break_even_utilization_pct

        elif c_1yr.monthly_net_savings > 0.0 and rate_1yr > 0.0:
            action = "1-Year CUD (100% Coverage)"
            action_type = "COMMIT"
            rec_target_cov = 1.0
            rec_rate = rate_1yr
            primary_reason = (
                f"Stable workload baseline. A 1-Year commitment secures ${c_1yr.monthly_net_savings:.2f}/mo net savings "
                f"at {int(rate_1yr*100)}% illustrative discount."
            )
            downside_warning = f"Break-even utilization is {c_1yr.break_even_utilization_pct}%."
            est_monthly_savings = c_1yr.monthly_net_savings
            est_annual_savings = c_1yr.annual_net_savings
            break_even = c_1yr.break_even_utilization_pct

        else:
            # Zero rates, zero coverage, or non-positive net savings
            action_type = "FLEXIBLE"
            est_monthly_savings = 0.0
            est_annual_savings = 0.0
            if rate_1yr <= 0.0 and rate_3yr <= 0.0:
                action = "Maintain Flexibility (0% CUD Discount Offers No Advantage)"
                primary_reason = (
                    "With CUD discount rates configured at 0%, entering a commitment provides zero pricing advantage "
                    "over on-demand while incurring an irrevocable contractual lock-in and forfeiting automatic SUD discounts. "
                    "Recommendation: Maintain 100% operational flexibility on On-Demand / SUD."
                )
                downside_warning = "Committing at 0% discount incurs 100% financial utilization liability for zero discount benefit."
                break_even = 100.0
            elif conserv_ratio <= 0.0 and (rate_3yr <= 0.0 or existing_cov > 0.0):
                action = "Maintain Flexibility (0% Commitment Coverage Configured)"
                primary_reason = (
                    "Conservative commitment target coverage is configured at 0%. No new commitment is established. "
                    "Workload continues operating on elastic On-Demand with automatic SUD discounts if eligible."
                )
                downside_warning = "Zero commitment coverage preserves full agility with $0 incremental savings."
                break_even = 100.0
            else:
                action = "Maintain Flexibility / SUD (Displacement Outweighs CUD)"
                primary_reason = (
                    f"At configured discount rates ({int(rate_1yr*100)}% 1-Yr, {int(rate_3yr*100)}% 3-Yr), incremental commitment savings "
                    "are zero or negative after accounting for displaced automatic SUD credits and utilization liability. "
                    "Maintain current operational posture."
                )
                downside_warning = "Entering a commitment with discount rates lower than or equal to active SUD credits generates net financial loss."
                break_even = round((1.0 - rate_1yr) * 100, 1) if rate_1yr > 0 else 100.0

    # Default fallback: Conservative 1-Year CUD only if net savings > 0 and rate > 0 and coverage > 0
    elif (
        c_conserv.monthly_net_savings > 0.0
        and rate_1yr > 0.0
        and conserv_ratio > 0.0
        and confidence.confidence_level in ["High", "Medium"]
    ):
        action = f"Conservative 1-Year CUD ({int(conserv_ratio*100)}% Coverage)"
        action_type = "COMMIT"
        rec_target_cov = conserv_ratio
        rec_rate = rate_1yr
        primary_reason = (
            f"Moderate stability (CV={cv:.2f}). A conservative commitment captures ${c_conserv.monthly_net_savings:.2f}/mo "
            f"in net savings while protecting against downside dips."
        )
        downside_warning = f"Break-even utilization is {c_conserv.break_even_utilization_pct}%."
        est_monthly_savings = c_conserv.monthly_net_savings
        est_annual_savings = c_conserv.annual_net_savings
        break_even = c_conserv.break_even_utilization_pct
    else:
        action_type = "FLEXIBLE"
        est_monthly_savings = 0.0
        est_annual_savings = 0.0
        if rate_1yr <= 0.0 and rate_3yr <= 0.0:
            action = "Maintain Flexibility (0% CUD Discount Offers No Advantage)"
            primary_reason = (
                "With CUD discount rates configured at 0%, entering a commitment provides zero pricing advantage over on-demand "
                "while incurring contract liability. Maintain 100% operational agility."
            )
            downside_warning = "Committing at 0% discount incurs 100% financial liability for zero discount benefit."
            break_even = 100.0
        elif conserv_ratio <= 0.0:
            action = "Maintain Flexibility (0% Commitment Coverage Configured)"
            primary_reason = (
                "Conservative target coverage is set to 0%. Workload operates on elastic On-Demand."
            )
            downside_warning = "Zero commitment coverage preserves full agility with $0 incremental savings."
            break_even = 100.0
        else:
            action = "Maintain Flexibility / SUD"
            primary_reason = (
                "Incremental financial benefit from new commitments is marginal after accounting for existing SUD credits "
                "and flexibility loss. Maintain current operational posture."
            )
            downside_warning = "Entering new commitments yields negative or negligible net ROI."
            break_even = round((1.0 - rate_1yr) * 100, 1) if rate_1yr > 0 else 100.0

    # Perform stress testing on the chosen commitment scenario
    stress_results = simulate_commitment_stress(
        baseline_uncovered_od=cud_res.uncovered_on_demand_cost,
        target_coverage_ratio=rec_target_cov if action_type == "COMMIT" else 1.0,  # stress test 1.0 if not committing to show why!
        discount_rate=rec_rate if action_type == "COMMIT" else rate_1yr,
        is_sud_eligible=cud_res.is_sud_eligible,
        avg_daily_hours=avg_hours,
        existing_commit_fee=cud_res.existing_commitment_cost,
        period_days=cud_res.period_days,
    )

    assumptions = [
        f"Discount rates applied: 1-Year = {int(rate_1yr*100)}%, 3-Year = {int(rate_3yr*100)}% (illustrative synthetic rates).",
        f"Break-even utilization for modeled commitment: {break_even}%.",
        "Assumes steady regional SKU pricing and single-instance workload continuity.",
        "Commitments are non-cancellable financial contracts in Google Cloud.",
    ]

    return WorkloadRecommendation(
        resource_id=cud_res.resource_id,
        project_id=cud_res.project_id,
        machine_type=cud_res.machine_type,
        recommended_action=action,
        action_type=action_type,
        confidence=confidence,
        break_even_utilization_pct=break_even,
        estimated_monthly_savings=round(est_monthly_savings, 2),
        estimated_annual_savings=round(est_annual_savings, 2),
        stress_test_results=stress_results,
        primary_reason=primary_reason,
        downside_risk_warning=downside_warning,
        assumptions_and_limitations=assumptions,
    )


def generate_portfolio_recommendations(
    df: pd.DataFrame,
    rates: Optional[Dict[str, float]] = None,
    enforce_gcp_family_policy: bool = True,
) -> List[WorkloadRecommendation]:
    """Generate recommendations across all workloads in the dataset."""
    recs: List[WorkloadRecommendation] = []
    for res_id, group in df.groupby("resource_id"):
        rec = generate_recommendation_for_workload(
            workload_df=group,
            rates=rates,
            enforce_gcp_family_policy=enforce_gcp_family_policy,
        )
        recs.append(rec)
    return recs
