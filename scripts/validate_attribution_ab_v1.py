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

    required_paths = [
        DATA / "attributed_label_log_1d.csv",
        DATA / "attributed_label_log_3d.csv",
        DATA / "attributed_label_log_7d.csv",
        DATA / "attributed_label_log_wide.csv",
        DATA / "ab_assignment_log.csv",
        DATA / "metasignal_integration_events.csv",
        Path("outputs/evidence/conversion_attribution_report.json"),
        Path("outputs/evidence/ab_simulation_results.json"),
        Path("outputs/evidence/metasignal_integration_events.json"),
        Path("outputs/reports/group6_attribution_ab_report_v1.json"),
    ]

    for path in required_paths:
        add(checks, f"exists::{path}", path.exists(), str(path))

    impressions = read_csv(DATA / "impression_log.csv") if (DATA / "impression_log.csv").exists() else []
    attr_3d = read_csv(DATA / "attributed_label_log_3d.csv") if (DATA / "attributed_label_log_3d.csv").exists() else []
    wide = read_csv(DATA / "attributed_label_log_wide.csv") if (DATA / "attributed_label_log_wide.csv").exists() else []
    assignments = read_csv(DATA / "ab_assignment_log.csv") if (DATA / "ab_assignment_log.csv").exists() else []
    msig_rows = read_csv(DATA / "metasignal_integration_events.csv") if (DATA / "metasignal_integration_events.csv").exists() else []

    add(
        checks,
        "attributed_3d_rows_equal_impressions",
        len(attr_3d) == len(impressions),
        {"attr_3d": len(attr_3d), "impressions": len(impressions)}
    )
    add(
        checks,
        "wide_rows_equal_impressions",
        len(wide) == len(impressions),
        {"wide": len(wide), "impressions": len(impressions)}
    )
    add(checks, "assignments_positive", len(assignments) > 0, len(assignments))
    add(checks, "metasignal_rows_positive", len(msig_rows) > 0, len(msig_rows))

    if wide:
        cols = list(wide[0].keys())
        for col in [
            "attributed_purchase_1d",
            "attributed_purchase_3d",
            "attributed_purchase_7d",
            "net_revenue_1d",
            "net_revenue_3d",
            "net_revenue_7d",
        ]:
            add(checks, f"wide_column_exists::{col}", col in cols, cols)

    if assignments:
        variants = {row["variant"] for row in assignments}
        add(
            checks,
            "assignments_have_both_variants",
            variants == {"control_raw_ranker", "treatment_governed_rerank"},
            variants
        )

    if msig_rows:
        required_cols = [
            "event_id",
            "source_project",
            "experiment_id",
            "variant",
            "metric_name",
            "metric_value",
            "unit_of_analysis",
            "timestamp",
            "claim_boundary",
        ]
        cols = list(msig_rows[0].keys())
        for col in required_cols:
            add(checks, f"metasignal_col_exists::{col}", col in cols, cols)

    attribution_path = Path("outputs/evidence/conversion_attribution_report.json")
    ab_path = Path("outputs/evidence/ab_simulation_results.json")
    msig_path = Path("outputs/evidence/metasignal_integration_events.json")
    group_path = Path("outputs/reports/group6_attribution_ab_report_v1.json")

    attribution = json.loads(attribution_path.read_text(encoding="utf-8")) if attribution_path.exists() else {}
    ab = json.loads(ab_path.read_text(encoding="utf-8")) if ab_path.exists() else {}
    msig = json.loads(msig_path.read_text(encoding="utf-8")) if msig_path.exists() else {}
    group = json.loads(group_path.read_text(encoding="utf-8")) if group_path.exists() else {}

    add(checks, "attribution_status_pass", attribution.get("status") == "pass", attribution.get("status"))
    add(
        checks,
        "attribution_windows_1_3_7",
        attribution.get("attribution_windows_days") == [1, 3, 7],
        attribution.get("attribution_windows_days")
    )
    add(
        checks,
        "attribution_has_window_summaries",
        set(attribution.get("window_summaries", {}).keys()) == {"1", "3", "7"},
        list(attribution.get("window_summaries", {}).keys())
    )

    add(checks, "ab_status_pass", ab.get("status") == "pass", ab.get("status"))
    add(checks, "ab_has_control_treatment", "control" in ab and "treatment" in ab, list(ab.keys()))
    add(checks, "ab_primary_metric_net_rps", ab.get("primary_metric") == "net_revenue_per_session", ab.get("primary_metric"))
    add(checks, "ab_decision_present", ab.get("decision") in {"SHIP_SIMULATED", "HOLD_SIMULATED"}, ab.get("decision"))
    add(
        checks,
        "ab_guardrails_present",
        isinstance(ab.get("guardrails"), dict) and len(ab.get("guardrails", {})) >= 4,
        ab.get("guardrails")
    )
    add(checks, "ab_has_claim_boundary", "not an online experiment" in ab.get("interpretation", ""), ab.get("interpretation"))
    add(checks, "msig_status_pass", msig.get("status") == "pass", msig.get("status"))
    add(checks, "msig_event_count_positive", int(msig.get("event_count", 0)) > 0, msig.get("event_count"))
    add(checks, "msig_claim_boundary_present", "no live MetaSignal ingestion" in msig.get("claim_boundary", ""), msig.get("claim_boundary"))
    add(checks, "group_status_pass", group.get("status") == "pass", group.get("status"))

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_attribution_ab_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 6 delayed attribution windows, return-adjusted revenue, offline A/B simulation, guardrails, and MetaSignal-compatible event export."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "attribution_ab_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_attribution_ab_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
