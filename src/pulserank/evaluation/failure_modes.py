from __future__ import annotations

from typing import Any


def scenario(
    scenario_id: str,
    name: str,
    trigger: str,
    detection_signal: str,
    action: str,
    final_state: str,
    severity: str,
    artifact_reference: str,
    interview_point: str,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "name": name,
        "trigger": trigger,
        "detection_signal": detection_signal,
        "action": action,
        "final_state": final_state,
        "severity": severity,
        "artifact_reference": artifact_reference,
        "interview_point": interview_point,
    }


def pulserank_failure_scenarios() -> list[dict[str, Any]]:
    return [
        scenario(
            "F-01",
            "missing_display_rank",
            "impression_log has impressions without display_rank",
            "display_rank_coverage < 1.0",
            "block IPS correction and fail corpus validation",
            "BLOCKED",
            "critical",
            "outputs/evidence/schema_manifest.json",
            "Position-bias correction is impossible without logged display rank.",
        ),
        scenario(
            "F-02",
            "invalid_display_rank_range",
            "display_rank falls outside 1..10",
            "display_rank_range_1_to_10 validation fails",
            "quarantine malformed impressions",
            "BLOCKED",
            "critical",
            "outputs/validation/corpus_validation_v1.json",
            "Ranking logs need strict position semantics before evaluation.",
        ),
        scenario(
            "F-03",
            "low_candidate_recall",
            "hybrid candidate recall drops below popularity baseline by large margin",
            "candidate_recall_report.hybrid_recall_at_100 regression",
            "fallback to popularity + content hybrid and mark retrieval quality degraded",
            "RECOVERED_WITH_FALLBACK",
            "high",
            "outputs/evidence/candidate_recall_report.json",
            "Items not retrieved cannot be ranked; retrieval recall is a first-order gate.",
        ),
        scenario(
            "F-04",
            "temporal_split_not_used",
            "ranking baseline evaluated with random split",
            "offline_eval_log.split_strategy != temporal_holdout",
            "reject evaluation and require temporal holdout rerun",
            "BLOCKED",
            "critical",
            "outputs/evidence/offline_eval_log.json",
            "Random split can leak future popularity and behavior into ranking evaluation.",
        ),
        scenario(
            "F-05",
            "propensity_missing_or_zero",
            "rank-level click propensity missing or zero",
            "propensity_by_rank_report has zero clipped_propensity",
            "apply propensity floor and rerun IPS validation",
            "RECOVERED_WITH_CLIPPING",
            "critical",
            "outputs/evidence/propensity_by_rank_report.json",
            "IPS can explode when propensities are too small; clipping controls variance.",
        ),
        scenario(
            "F-06",
            "ips_variance_explosion",
            "IPS weight distribution has extreme max weight",
            "bias_correction_report.weight_distribution.max_weight over cap",
            "increase propensity floor, report limitation, and mark estimate high-variance",
            "REVIEW_REQUIRED",
            "high",
            "outputs/evidence/bias_correction_report.json",
            "Debiasing improves validity but can increase variance.",
        ),
        scenario(
            "F-07",
            "seller_exposure_concentration",
            "top-10 exposure over-concentrates sellers",
            "diversity_guardrail_log.after_rerank.seller_gini_at_10 > threshold",
            "hold promotion and rerun governed reranker with stricter seller cap",
            "HOLD",
            "high",
            "outputs/evidence/diversity_guardrail_log.json",
            "Marketplace ranking is not just relevance; supply-side exposure matters.",
        ),
        scenario(
            "F-08",
            "category_monoculture",
            "top-10 dominated by one category",
            "mean_category_max_share_at_10 crosses review band",
            "apply MMR/category constraint and monitor relevance tradeoff",
            "REVIEW_REQUIRED",
            "medium",
            "outputs/evidence/diversity_guardrail_log.json",
            "Diversity constraints are a deliberate relevance-vs-marketplace-health tradeoff.",
        ),
        scenario(
            "F-09",
            "delayed_attribution_window_missing",
            "only same-session click labels available",
            "conversion_attribution_report missing 1/3/7-day windows",
            "block A/B simulation until delayed attribution is generated",
            "BLOCKED",
            "critical",
            "outputs/evidence/conversion_attribution_report.json",
            "CTR alone is insufficient; purchases arrive after impression time.",
        ),
        scenario(
            "F-10",
            "return_adjustment_missing",
            "attributed revenue ignores returns",
            "net_revenue_after_returns missing from attribution report",
            "block primary metric readout and regenerate return-adjusted labels",
            "BLOCKED",
            "high",
            "outputs/evidence/conversion_attribution_report.json",
            "Revenue/session should be adjusted for returns in marketplace evaluation.",
        ),
        scenario(
            "F-11",
            "guardrail_breach_despite_primary_lift",
            "treatment improves net RPS but violates seller/category/return guardrail",
            "ab_simulation_results.guardrails_pass is false",
            "decision becomes HOLD_SIMULATED",
            "HOLD",
            "high",
            "outputs/evidence/ab_simulation_results.json",
            "A positive primary metric is not enough to ship if guardrails fail.",
        ),
        scenario(
            "F-12",
            "metasignal_export_schema_mismatch",
            "PulseRank event export missing metric_name or unit_of_analysis",
            "metasignal_integration_events schema validation fails",
            "block integration export and regenerate schema-compatible events",
            "BLOCKED",
            "medium",
            "outputs/evidence/metasignal_integration_events.json",
            "Portfolio integration should be schema-compatible, not hand-wavy.",
        ),
        scenario(
            "F-13",
            "cold_start_items_starved",
            "new or sparse-history items receive no exposure",
            "cold_start_events count below expected and catalog coverage low",
            "trigger content-based fallback and exploration allocation",
            "RECOVERED_WITH_FALLBACK",
            "medium",
            "outputs/evidence/corpus_summary.json",
            "Ranking systems need cold-start fallback, not only popularity loops.",
        ),
        scenario(
            "F-14",
            "exploration_over_allocation",
            "exploration traffic exceeds configured cap",
            "exploration share above hard_cap",
            "throttle exploration and prioritize deterministic ranking",
            "THROTTLED",
            "medium",
            "configs/policy_config.json",
            "Exploration is useful but must be bounded by policy.",
        ),
        scenario(
            "F-15",
            "offline_ab_overclaim",
            "offline simulation interpreted as real online experiment",
            "claim boundary check detects online A/B claim",
            "rewrite report to clarify offline simulation only",
            "CLAIM_CORRECTED",
            "critical",
            "docs/PULSERANK_ATTRIBUTION_AB.md",
            "Truth boundary is part of the project quality, not a weakness.",
        ),
    ]


def aggregate_failure_states(scenarios: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in scenarios:
        state = row["final_state"]
        counts[state] = counts.get(state, 0) + 1
    return counts


def aggregate_severity(scenarios: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in scenarios:
        sev = row["severity"]
        counts[sev] = counts.get(sev, 0) + 1
    return counts
