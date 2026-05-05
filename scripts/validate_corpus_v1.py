from __future__ import annotations

import csv
import json
from pathlib import Path


DATA = Path("data/processed")
OUT = Path("outputs/validation")

REQUIRED_FILES = {
    "user_sessions": DATA / "user_sessions.csv",
    "item_catalog": DATA / "item_catalog.csv",
    "impression_log": DATA / "impression_log.csv",
    "purchase_log": DATA / "purchase_log.csv",
    "attributed_label_log": DATA / "attributed_label_log.csv",
    "exploration_log": DATA / "exploration_log.csv",
    "cold_start_events": DATA / "cold_start_events.csv",
}

REQUIRED_COLUMNS = {
    "user_sessions": ["session_id", "user_id", "timestamp_start", "timestamp_end", "device_type", "num_events", "category_affinity", "price_sensitivity_bucket", "simulated_day"],
    "item_catalog": ["item_id", "category_l1", "category_l2", "price", "seller_id", "content_embedding", "popularity_rank", "freshness_score", "available", "created_at"],
    "impression_log": ["impression_id", "session_id", "user_id", "item_id", "display_rank", "recommendation_stage", "ranking_variant_id", "exploration_flag", "cold_start_flag", "timestamp", "clicked", "added_to_cart", "position_propensity"],
    "purchase_log": ["purchase_id", "session_id", "user_id", "item_id", "impression_id", "purchase_timestamp", "revenue", "returned", "return_timestamp"],
    "attributed_label_log": ["impression_id", "session_id", "item_id", "attributed_purchase", "attribution_window_days", "attribution_timestamp", "attributed_revenue", "attributed_return", "data_gap_flag"],
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(checks: list[dict], name: str, passed: bool, observed) -> None:
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def main() -> None:
    checks: list[dict] = []

    for name, path in REQUIRED_FILES.items():
        add(checks, f"file_exists::{name}", path.exists(), str(path))

    tables = {}
    for name, path in REQUIRED_FILES.items():
        if path.exists():
            rows = read_csv(path)
            tables[name] = rows
            add(checks, f"row_count_positive::{name}", len(rows) > 0, len(rows))
        else:
            tables[name] = []

    for table_name, cols in REQUIRED_COLUMNS.items():
        rows = tables.get(table_name, [])
        actual_cols = list(rows[0].keys()) if rows else []
        for col in cols:
            add(checks, f"column_exists::{table_name}.{col}", col in actual_cols, actual_cols)

    impressions = tables["impression_log"]
    sessions = tables["user_sessions"]
    items = tables["item_catalog"]
    purchases = tables["purchase_log"]
    attributed = tables["attributed_label_log"]

    if impressions:
        ranks = [int(x["display_rank"]) for x in impressions if str(x.get("display_rank", "")).strip()]
        add(checks, "display_rank_complete", len(ranks) == len(impressions), {"ranks": len(ranks), "impressions": len(impressions)})
        add(checks, "display_rank_range_1_to_10", all(1 <= r <= 10 for r in ranks), {"min": min(ranks), "max": max(ranks)})
        add(checks, "impression_has_position_propensity", all(float(x["position_propensity"]) > 0 for x in impressions), "positive propensities")
        add(checks, "click_labels_binary", all(x["clicked"] in {"0", "1"} for x in impressions), "clicked in {0,1}")
        add(checks, "atc_labels_binary", all(x["added_to_cart"] in {"0", "1"} for x in impressions), "added_to_cart in {0,1}")

    add(checks, "session_count_minimum", len(sessions) >= 2500, len(sessions))
    add(checks, "item_count_minimum", len(items) >= 500, len(items))
    add(checks, "impression_count_minimum", len(impressions) >= 25000, len(impressions))
    add(checks, "purchase_count_minimum", len(purchases) >= 200, len(purchases))
    add(checks, "attributed_rows_equal_impressions", len(attributed) == len(impressions), {"attributed": len(attributed), "impressions": len(impressions)})

    summary_path = Path("outputs/evidence/corpus_summary.json")
    schema_path = Path("outputs/evidence/schema_manifest.json")
    samples_path = Path("outputs/evidence/corpus_samples.json")
    report_path = Path("outputs/reports/group1_corpus_report_v1.json")

    for path in [summary_path, schema_path, samples_path, report_path]:
        add(checks, f"artifact_exists::{path}", path.exists(), str(path))

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    add(checks, "summary_status_pass", summary.get("status") == "pass", summary.get("status"))
    add(checks, "summary_display_rank_coverage_1", summary.get("display_rank_coverage") == 1.0, summary.get("display_rank_coverage"))
    add(checks, "summary_has_revenue", float(summary.get("revenue", 0)) > 0, summary.get("revenue"))

    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    add(checks, "schema_manifest_status_pass", schema.get("status") == "pass", schema.get("status"))
    add(checks, "schema_mentions_display_rank", schema.get("non_negotiable_field") == "impression_log.display_rank", schema.get("non_negotiable_field"))

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_corpus_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 1 deterministic marketplace corpus and confirms every impression has display_rank for future IPS correction."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "corpus_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_corpus_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
