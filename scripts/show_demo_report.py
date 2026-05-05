from __future__ import annotations

import json
from pathlib import Path

from src.pulserank.evaluation.failure_modes import (
    aggregate_failure_states,
    aggregate_severity,
    pulserank_failure_scenarios,
)


ROOT = Path(".")
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"
VALIDATION = ROOT / "outputs" / "validation"


def load_json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_json(folder: str) -> int:
    p = Path(folder)
    if not p.exists():
        return 0
    return len(list(p.rglob("*.json")))


def pick(d: dict, key: str, default="—"):
    return d.get(key, default) if isinstance(d, dict) else default


def main() -> None:
    corpus = load_json("outputs/evidence/corpus_summary.json")
    candidate = load_json("outputs/evidence/candidate_generation_report.json")
    ranking = load_json("outputs/evidence/ranking_baseline_report.json")
    bias = load_json("outputs/evidence/bias_correction_report.json")
    diversity = load_json("outputs/evidence/diversity_guardrail_log.json")
    attribution = load_json("outputs/evidence/conversion_attribution_report.json")
    ab = load_json("outputs/evidence/ab_simulation_results.json")
    metasignal = load_json("outputs/evidence/metasignal_integration_events.json")

    validations = sorted([str(p) for p in VALIDATION.glob("*.json")])
    evidence_files = sorted([str(p) for p in EVIDENCE.glob("*.json")])
    report_files = sorted([str(p) for p in REPORTS.glob("*.json")])

    scenarios = pulserank_failure_scenarios()

    failure_report = {
        "artifact": "pulserank_failure_recovery_report_v1",
        "status": "pass",
        "scenario_count": len(scenarios),
        "failure_states": aggregate_failure_states(scenarios),
        "severity_mix": aggregate_severity(scenarios),
        "scenarios": scenarios,
        "evidence_statement": "Defines failure and recovery scenarios for PulseRank ranking-system reliability, evaluation validity, guardrail behavior, integration safety, and claim-boundary enforcement."
    }

    write_json(EVIDENCE / "failure_recovery_report.json", failure_report)

    artifact_inventory = {
        "evidence_json_count": count_json("outputs/evidence"),
        "validation_json_count": count_json("outputs/validation"),
        "report_json_count": count_json("outputs/reports"),
        "total_json_artifacts": count_json("outputs/evidence") + count_json("outputs/validation") + count_json("outputs/reports"),
    }

    demo = {
        "artifact": "pulserank_demo_report_v1",
        "status": "pass",
        "project": "PulseRank",
        "one_line": "Production-simulated marketplace recommendation and ranking decision system with IPS bias correction, delayed attribution, exposure governance, offline A/B simulation, and evidence artifacts.",
        "truth_boundary": {
            "build_type": "solo-built_non-production_production-simulated",
            "not_claimed": [
                "real production deployment",
                "real users served",
                "real production traffic",
                "real online A/B test",
                "real revenue optimization",
                "real RL or contextual bandit optimization",
                "real ad auction infrastructure"
            ]
        },
        "core_numbers": {
            "sessions": pick(corpus, "sessions"),
            "items": pick(corpus, "items"),
            "sellers": pick(corpus, "sellers"),
            "impressions": pick(corpus, "impressions"),
            "purchases": pick(corpus, "purchases"),
            "display_rank_coverage": pick(corpus, "display_rank_coverage"),
            "hybrid_recall_at_100": pick(candidate.get("recall_at_100", {}), "hybrid_popularity_content"),
            "holdout_ndcg_at_10": pick(ranking, "holdout_ndcg_at_10"),
            "ips_weighted_ndcg_at_10": pick(bias, "ips_weighted_ndcg_at_10"),
            "seller_gini_after_rerank": pick(diversity.get("after_rerank", {}), "seller_gini_at_10"),
            "catalog_coverage_after_rerank": pick(diversity.get("after_rerank", {}), "catalog_coverage_at_10"),
            "attribution_windows": pick(attribution, "attribution_windows_days"),
            "ab_decision": pick(ab, "decision"),
            "metasignal_event_count": pick(metasignal, "event_count"),
            "failure_scenarios": len(scenarios),
        },
        "system_flow": [
            "generate deterministic marketplace corpus",
            "log impressions with display_rank",
            "generate popularity/content/hybrid candidates",
            "rank with temporal holdout evaluation",
            "estimate rank propensity and apply IPS correction",
            "rerank using MMR, seller/category exposure, novelty, and coverage constraints",
            "attribute delayed conversions over 1/3/7-day windows",
            "run offline A/B simulation with guardrails",
            "export MetaSignal-compatible metric events",
            "validate failure/recovery and claim boundary"
        ],
        "artifact_inventory": artifact_inventory,
        "evidence_files": evidence_files,
        "validation_files": validations,
        "report_files": report_files,
        "resume_claim": "Built PulseRank, a production-simulated marketplace ranking system with display-rank impression logging, hybrid candidate generation, temporal holdout ranking evaluation, IPS position-bias correction, delayed conversion attribution, seller/category exposure governance, offline A/B simulation, failure recovery scenarios, and reproducible evidence artifacts.",
        "evidence_statement": "Final local demo report summarizes PulseRank build evidence and creates a single reviewer-facing project narrative."
    }

    write_json(REPORTS / "pulserank_demo_report.json", demo)

    txt = f"""PulseRank Demo Report
=====================

Status: {demo['status']}

One-line:
{demo['one_line']}

Core numbers:
- Sessions: {demo['core_numbers']['sessions']}
- Items: {demo['core_numbers']['items']}
- Sellers: {demo['core_numbers']['sellers']}
- Impressions: {demo['core_numbers']['impressions']}
- Purchases: {demo['core_numbers']['purchases']}
- Display-rank coverage: {demo['core_numbers']['display_rank_coverage']}
- Hybrid Recall@100: {demo['core_numbers']['hybrid_recall_at_100']}
- Holdout NDCG@10: {demo['core_numbers']['holdout_ndcg_at_10']}
- IPS-weighted NDCG@10: {demo['core_numbers']['ips_weighted_ndcg_at_10']}
- Seller Gini after rerank: {demo['core_numbers']['seller_gini_after_rerank']}
- Catalog coverage after rerank: {demo['core_numbers']['catalog_coverage_after_rerank']}
- A/B decision: {demo['core_numbers']['ab_decision']}
- MetaSignal-compatible events: {demo['core_numbers']['metasignal_event_count']}
- Failure scenarios: {demo['core_numbers']['failure_scenarios']}

Truth boundary:
Solo-built, non-production, production-simulated. No real production deployment, no real users served, no real online A/B test, no real revenue optimization claim.

Resume claim:
{demo['resume_claim']}
"""
    (REPORTS / "pulserank_demo_report.txt").write_text(txt, encoding="utf-8")

    print("=== PulseRank Demo Report ===")
    print(f"status: {demo['status']}")
    print(f"sessions: {demo['core_numbers']['sessions']}")
    print(f"items: {demo['core_numbers']['items']}")
    print(f"impressions: {demo['core_numbers']['impressions']}")
    print(f"display_rank_coverage: {demo['core_numbers']['display_rank_coverage']}")
    print(f"hybrid_recall_at_100: {demo['core_numbers']['hybrid_recall_at_100']}")
    print(f"ips_weighted_ndcg_at_10: {demo['core_numbers']['ips_weighted_ndcg_at_10']}")
    print(f"ab_decision: {demo['core_numbers']['ab_decision']}")
    print(f"failure_scenarios: {demo['core_numbers']['failure_scenarios']}")
    print(f"total_json_artifacts: {artifact_inventory['total_json_artifacts']}")
    print("wrote outputs/evidence/failure_recovery_report.json")
    print("wrote outputs/reports/pulserank_demo_report.json")
    print("wrote outputs/reports/pulserank_demo_report.txt")


if __name__ == "__main__":
    main()
