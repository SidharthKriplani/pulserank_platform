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

    ips_path = DATA / "ips_training_examples.csv"
    propensity_path = Path("outputs/evidence/propensity_by_rank_report.json")
    bias_path = Path("outputs/evidence/bias_correction_report.json")
    group_path = Path("outputs/reports/group4_bias_correction_report_v1.json")

    for path in [ips_path, propensity_path, bias_path, group_path]:
        add(checks, f"exists::{path}", path.exists(), str(path))

    ips_rows = read_csv(ips_path) if ips_path.exists() else []
    impressions = read_csv(DATA / "impression_log.csv") if (DATA / "impression_log.csv").exists() else []

    add(checks, "ips_rows_positive", len(ips_rows) > 0, len(ips_rows))
    add(
        checks,
        "ips_rows_equal_impressions",
        len(ips_rows) == len(impressions),
        {"ips_rows": len(ips_rows), "impressions": len(impressions)}
    )

    if ips_rows:
        required_cols = [
            "impression_id",
            "session_id",
            "item_id",
            "display_rank",
            "split",
            "clicked",
            "attributed_purchase",
            "empirical_propensity",
            "clipped_propensity",
            "ips_weight",
            "propensity_was_clipped"
        ]

        cols = list(ips_rows[0].keys())
        for col in required_cols:
            add(checks, f"column_exists::{col}", col in cols, cols)

        ranks = [int(r["display_rank"]) for r in ips_rows]
        add(checks, "display_rank_range_1_10", min(ranks) == 1 and max(ranks) == 10, {"min": min(ranks), "max": max(ranks)})

        weights = [float(r["ips_weight"]) for r in ips_rows]
        add(checks, "ips_weight_positive", all(w > 0 for w in weights), {"min": min(weights), "max": max(weights)})

        splits = {r["split"] for r in ips_rows}
        add(checks, "has_train_and_holdout", splits == {"train", "holdout"}, sorted(list(splits)))

    propensity = json.loads(propensity_path.read_text(encoding="utf-8")) if propensity_path.exists() else {}
    bias = json.loads(bias_path.read_text(encoding="utf-8")) if bias_path.exists() else {}
    group = json.loads(group_path.read_text(encoding="utf-8")) if group_path.exists() else {}

    add(checks, "propensity_status_pass", propensity.get("status") == "pass", propensity.get("status"))
    add(checks, "propensity_floor_001", float(propensity.get("propensity_floor", 0)) == 0.01, propensity.get("propensity_floor"))

    prop_rows = propensity.get("propensity_by_rank", [])
    add(checks, "propensity_has_10_ranks", len(prop_rows) == 10, len(prop_rows))
    if prop_rows:
        ranks = [int(r["display_rank"]) for r in prop_rows]
        add(checks, "propensity_ranks_1_to_10", ranks == list(range(1, 11)), ranks)
        add(checks, "propensity_clicks_nonzero", sum(int(r["clicks"]) for r in prop_rows) > 0, sum(int(r["clicks"]) for r in prop_rows))
        add(checks, "propensity_weights_positive", all(float(r["ips_weight"]) > 0 for r in prop_rows), prop_rows)

    add(checks, "bias_status_pass", bias.get("status") == "pass", bias.get("status"))
    add(checks, "bias_method_ips", bias.get("method") == "inverse_propensity_scoring", bias.get("method"))
    add(checks, "bias_has_raw_ndcg", "raw_ndcg_at_10" in bias, bias.get("raw_ndcg_at_10"))
    add(checks, "bias_has_ips_ndcg", "ips_weighted_ndcg_at_10" in bias, bias.get("ips_weighted_ndcg_at_10"))
    add(checks, "bias_has_delta", "bias_delta_ips_minus_raw" in bias, bias.get("bias_delta_ips_minus_raw"))
    add(checks, "bias_evaluated_sessions_positive", int(bias.get("evaluated_holdout_purchase_sessions", 0)) > 0, bias.get("evaluated_holdout_purchase_sessions"))
    add(checks, "bias_has_assumption_limitations", len(bias.get("assumption_limitations", [])) >= 3, bias.get("assumption_limitations"))
    add(checks, "group_status_pass", group.get("status") == "pass", group.get("status"))

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_bias_correction_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 4 IPS position-bias correction, propensity-by-rank report, clipped weights, and raw vs IPS-weighted NDCG evidence."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "bias_correction_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_bias_correction_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
