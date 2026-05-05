from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DATA = Path("data/processed")
OUT = Path("outputs/validation")


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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(checks: list[dict], name: str, passed: bool, observed) -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "observed": json_safe(observed)
    })


def main() -> None:
    checks = []

    reranked_path = DATA / "reranked_lists.csv"
    audit_path = DATA / "rerank_audit_log.csv"
    guardrail_path = Path("outputs/evidence/diversity_guardrail_log.json")
    diversity_path = Path("outputs/evidence/diversity_report.json")
    coverage_path = Path("outputs/evidence/catalog_coverage_report.json")
    group_path = Path("outputs/reports/group5_reranking_constraints_report_v1.json")

    for path in [reranked_path, audit_path, guardrail_path, diversity_path, coverage_path, group_path]:
        add(checks, f"exists::{path}", path.exists(), str(path))

    reranked = read_csv(reranked_path) if reranked_path.exists() else []
    audit = read_csv(audit_path) if audit_path.exists() else []
    sessions = read_csv(DATA / "user_sessions.csv") if (DATA / "user_sessions.csv").exists() else []

    add(checks, "reranked_rows_positive", len(reranked) > 0, len(reranked))
    add(
        checks,
        "reranked_rows_10_per_session",
        len(reranked) == len(sessions) * 10,
        {"reranked": len(reranked), "sessions_x10": len(sessions) * 10}
    )
    add(
        checks,
        "audit_rows_match_sessions",
        len(audit) == len(sessions),
        {"audit": len(audit), "sessions": len(sessions)}
    )

    if reranked:
        required_cols = [
            "session_id",
            "item_id",
            "reranked_position",
            "pre_rerank_score",
            "label_purchase_in_session",
            "category_l1",
            "seller_id",
            "simulated_day",
            "split"
        ]
        cols = list(reranked[0].keys())
        for col in required_cols:
            add(checks, f"column_exists::{col}", col in cols, cols)

        positions = [int(x["reranked_position"]) for x in reranked]
        add(
            checks,
            "reranked_position_range_1_10",
            min(positions) == 1 and max(positions) == 10,
            {"min": min(positions), "max": max(positions)}
        )

    if audit:
        add(
            checks,
            "audit_has_unique_sellers",
            all(int(x["unique_sellers"]) > 0 for x in audit),
            "unique_sellers positive"
        )
        add(
            checks,
            "audit_has_unique_categories",
            all(int(x["unique_categories"]) > 0 for x in audit),
            "unique_categories positive"
        )

    guardrail = json.loads(guardrail_path.read_text(encoding="utf-8")) if guardrail_path.exists() else {}
    diversity = json.loads(diversity_path.read_text(encoding="utf-8")) if diversity_path.exists() else {}
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    group = json.loads(group_path.read_text(encoding="utf-8")) if group_path.exists() else {}

    add(checks, "guardrail_status_valid", guardrail.get("status") in {"pass", "review", "fail"}, guardrail.get("status"))
    add(checks, "guardrail_has_constraints", "constraints" in guardrail, guardrail.get("constraints"))
    add(checks, "guardrail_has_before_after", "before_rerank" in guardrail and "after_rerank" in guardrail, list(guardrail.keys()))
    add(checks, "seller_gini_threshold_recorded", guardrail.get("constraints", {}).get("seller_gini_threshold") == 0.65, guardrail.get("constraints", {}))
    add(checks, "mmr_lambda_recorded", guardrail.get("constraints", {}).get("mmr_lambda") == 0.70, guardrail.get("constraints", {}))

    after = guardrail.get("after_rerank", {})
    add(checks, "after_seller_gini_present", "seller_gini_at_10" in after, after)
    add(checks, "after_ild_present", "mean_ild_at_10" in after, after)
    add(checks, "after_coverage_present", "catalog_coverage_at_10" in after, after)

    add(checks, "diversity_status_pass", diversity.get("status") == "pass", diversity.get("status"))
    add(checks, "diversity_sessions_positive", int(diversity.get("sessions_reranked", 0)) > 0, diversity.get("sessions_reranked"))
    add(checks, "coverage_status_pass", coverage.get("status") == "pass", coverage.get("status"))
    add(checks, "coverage_values_present", "reranked_catalog_coverage_at_10" in coverage, coverage)
    add(checks, "group_status_pass", group.get("status") == "pass", group.get("status"))

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_reranking_constraints_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 5 reranking constraints, seller/category exposure governance, diversity guardrail log, catalog coverage, and audit trail."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "reranking_constraints_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_reranking_constraints_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
