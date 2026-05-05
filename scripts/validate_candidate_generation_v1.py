from __future__ import annotations

import csv
import json
from pathlib import Path


DATA = Path("data/processed")
OUT = Path("outputs/validation")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(checks: list[dict], name: str, passed: bool, observed) -> None:
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def main() -> None:
    checks = []

    candidate_path = DATA / "candidate_sets.csv"
    report_path = Path("outputs/evidence/candidate_generation_report.json")
    recall_path = Path("outputs/evidence/candidate_recall_report.json")
    sample_path = Path("outputs/evidence/candidate_samples.json")
    group_path = Path("outputs/reports/group2_candidate_generation_report_v1.json")

    for path in [candidate_path, report_path, recall_path, sample_path, group_path]:
        add(checks, f"exists::{path}", path.exists(), str(path))

    candidate_rows = read_csv(candidate_path) if candidate_path.exists() else []
    sessions = read_csv(DATA / "user_sessions.csv") if (DATA / "user_sessions.csv").exists() else []

    add(checks, "candidate_rows_match_sessions", len(candidate_rows) == len(sessions), {"candidate_rows": len(candidate_rows), "sessions": len(sessions)})
    add(checks, "candidate_rows_positive", len(candidate_rows) > 0, len(candidate_rows))

    if candidate_rows:
        sizes = [int(row["candidate_set_size"]) for row in candidate_rows]
        add(checks, "candidate_set_size_100", all(x == 100 for x in sizes), {"min": min(sizes), "max": max(sizes)})
        parsed_first = json.loads(candidate_rows[0]["candidate_item_ids"])
        add(checks, "candidate_ids_parseable", isinstance(parsed_first, list) and len(parsed_first) == 100, len(parsed_first))
        add(checks, "top10_parseable", len(json.loads(candidate_rows[0]["top_10_candidate_item_ids"])) == 10, candidate_rows[0]["top_10_candidate_item_ids"])

    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    recall = json.loads(recall_path.read_text(encoding="utf-8")) if recall_path.exists() else {}

    add(checks, "report_status_pass", report.get("status") == "pass", report.get("status"))
    add(checks, "report_has_three_generators", len(report.get("generators", [])) == 3, report.get("generators"))
    add(checks, "report_sessions_scored_positive", int(report.get("sessions_scored", 0)) > 0, report.get("sessions_scored"))
    add(checks, "report_recall_values_present", set(report.get("recall_at_100", {}).keys()) == {"popularity", "content_similarity", "hybrid_popularity_content"}, report.get("recall_at_100"))

    hybrid_recall = float(recall.get("hybrid_recall_at_100", 0))
    add(checks, "hybrid_recall_nonzero", hybrid_recall > 0, hybrid_recall)
    add(checks, "recall_evaluated_sessions_nonzero", int(recall.get("evaluated_sessions", 0)) > 0, recall.get("evaluated_sessions"))

    coverage_values = report.get("catalog_coverage_at_100", {})
    add(checks, "hybrid_catalog_coverage_positive", float(coverage_values.get("hybrid_popularity_content", 0)) > 0, coverage_values)

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_candidate_generation_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 2 candidate generation artifacts, session-level candidate sets, recall evidence, and catalog coverage."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "candidate_generation_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_candidate_generation_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
