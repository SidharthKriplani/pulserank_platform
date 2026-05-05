from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OUT = Path("outputs/validation")


REQUIRED_ARTIFACTS = [
    "outputs/evidence/corpus_summary.json",
    "outputs/evidence/candidate_generation_report.json",
    "outputs/evidence/ranking_baseline_report.json",
    "outputs/evidence/bias_correction_report.json",
    "outputs/evidence/diversity_guardrail_log.json",
    "outputs/evidence/conversion_attribution_report.json",
    "outputs/evidence/ab_simulation_results.json",
    "outputs/evidence/metasignal_integration_events.json",
    "outputs/evidence/failure_recovery_report.json",
    "outputs/reports/pulserank_demo_report.json",
    "outputs/reports/pulserank_demo_report.txt",
]

REQUIRED_VALIDATIONS = [
    "outputs/validation/scaffold_validation_v1.json",
    "outputs/validation/corpus_validation_v1.json",
    "outputs/validation/candidate_generation_validation_v1.json",
    "outputs/validation/ranking_baseline_validation_v1.json",
    "outputs/validation/bias_correction_validation_v1.json",
    "outputs/validation/reranking_constraints_validation_v1.json",
    "outputs/validation/attribution_ab_validation_v1.json",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if str(type(value)) in {"<class 'dict_keys'>", "<class 'dict_values'>", "<class 'dict_items'>"}:
        return list(value)
    return value


def load(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def add(checks: list[dict], name: str, passed: bool, observed) -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "observed": json_safe(observed)
    })


def main() -> None:
    checks = []

    for path in REQUIRED_ARTIFACTS:
        add(checks, f"required_artifact_exists::{path}", Path(path).exists(), path)

    for path in REQUIRED_VALIDATIONS:
        add(checks, f"required_validation_exists::{path}", Path(path).exists(), path)
        if Path(path).exists():
            payload = load(path)
            add(checks, f"required_validation_pass::{path}", payload.get("status") == "pass", payload.get("status"))

    failure = load("outputs/evidence/failure_recovery_report.json")
    demo = load("outputs/reports/pulserank_demo_report.json")
    ab = load("outputs/evidence/ab_simulation_results.json")
    corpus = load("outputs/evidence/corpus_summary.json")
    bias = load("outputs/evidence/bias_correction_report.json")

    add(checks, "failure_status_pass", failure.get("status") == "pass", failure.get("status"))
    add(checks, "failure_scenario_count_min_15", int(failure.get("scenario_count", 0)) >= 15, failure.get("scenario_count"))
    add(checks, "failure_has_blocked_state", "BLOCKED" in failure.get("failure_states", {}), failure.get("failure_states"))
    add(checks, "demo_status_pass", demo.get("status") == "pass", demo.get("status"))
    add(checks, "demo_has_resume_claim", "resume_claim" in demo and "PulseRank" in demo.get("resume_claim", ""), demo.get("resume_claim"))
    add(checks, "demo_has_truth_boundary", "truth_boundary" in demo, demo.get("truth_boundary"))
    add(checks, "demo_mentions_ips", "IPS" in demo.get("one_line", "") or "IPS" in demo.get("resume_claim", ""), demo.get("one_line"))
    add(checks, "demo_ab_decision_matches_ab_report", demo.get("core_numbers", {}).get("ab_decision") == ab.get("decision"), {"demo": demo.get("core_numbers", {}).get("ab_decision"), "ab": ab.get("decision")})
    add(checks, "display_rank_coverage_one", float(corpus.get("display_rank_coverage", 0)) == 1.0, corpus.get("display_rank_coverage"))
    add(checks, "bias_report_method_ips", bias.get("method") == "inverse_propensity_scoring", bias.get("method"))

    txt_path = Path("outputs/reports/pulserank_demo_report.txt")
    txt = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    add(checks, "txt_report_contains_truth_boundary", "Truth boundary" in txt, txt[:200])
    add(checks, "txt_report_contains_resume_claim", "Resume claim" in txt, txt[:300])

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_demo_report_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates PulseRank Group 7 failure scenarios, final demo report, artifact inventory, validation chain, truth boundary, and resume claim."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "demo_report_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_demo_report_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
