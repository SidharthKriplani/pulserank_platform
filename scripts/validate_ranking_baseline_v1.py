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

    ranked_path = DATA / "ranked_lists.csv"
    ranking_path = Path("outputs/evidence/ranking_baseline_report.json")
    eval_path = Path("outputs/evidence/offline_eval_log.json")
    registry_path = Path("outputs/evidence/model_registry.json")
    group_path = Path("outputs/reports/group3_ranking_baseline_report_v1.json")

    for path in [ranked_path, ranking_path, eval_path, registry_path, group_path]:
        add(checks, f"exists::{path}", path.exists(), str(path))

    ranked = read_csv(ranked_path) if ranked_path.exists() else []
    sessions = read_csv(DATA / "user_sessions.csv") if (DATA / "user_sessions.csv").exists() else []

    add(checks, "ranked_rows_positive", len(ranked) > 0, len(ranked))
    add(
        checks,
        "ranked_rows_10_per_session",
        len(ranked) == len(sessions) * 10,
        {"ranked": len(ranked), "sessions_x10": len(sessions) * 10}
    )

    if ranked:
        required_cols = [
            "session_id",
            "item_id",
            "model_rank",
            "ranking_score",
            "label_purchase_in_session",
            "simulated_day",
            "split"
        ]
        cols = list(ranked[0].keys())
        for col in required_cols:
            add(checks, f"column_exists::{col}", col in cols, cols)

        ranks = [int(x["model_rank"]) for x in ranked]
        add(
            checks,
            "model_rank_range_1_to_10",
            min(ranks) == 1 and max(ranks) == 10,
            {"min": min(ranks), "max": max(ranks)}
        )

        scores = [float(x["ranking_score"]) for x in ranked]
        add(
            checks,
            "ranking_scores_in_0_1",
            all(0 <= s <= 1 for s in scores),
            {"min": min(scores), "max": max(scores)}
        )

        splits = {x["split"] for x in ranked}
        add(checks, "has_train_and_holdout", splits == {"train", "holdout"}, splits)

    ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else {}
    offline = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}

    add(checks, "ranking_status_pass", ranking.get("status") == "pass", ranking.get("status"))
    add(checks, "temporal_split_enforced", ranking.get("temporal_split_enforced") is True, ranking.get("temporal_split_enforced"))
    add(checks, "holdout_sessions_positive", int(ranking.get("holdout_sessions", 0)) > 0, ranking.get("holdout_sessions"))
    add(checks, "holdout_ndcg_present", "holdout_ndcg_at_10" in ranking, ranking.get("holdout_ndcg_at_10"))
    add(checks, "seller_gini_present", "seller_gini_at_10" in ranking, ranking.get("seller_gini_at_10"))

    add(checks, "offline_status_pass", offline.get("status") == "pass", offline.get("status"))
    add(checks, "offline_temporal_holdout", offline.get("split_strategy") == "temporal_holdout", offline.get("split_strategy"))

    expected_metric_stack = {
        "ndcg_at_10",
        "mrr",
        "precision_at_10",
        "recall_at_10",
        "map_at_10",
        "seller_gini_at_10"
    }
    actual_metric_stack = set(offline.get("ranking_metric_stack", []))
    add(
        checks,
        "offline_metric_stack_complete",
        expected_metric_stack.issubset(actual_metric_stack),
        {
            "expected": expected_metric_stack,
            "actual": actual_metric_stack
        }
    )

    add(checks, "registry_status_pass", registry.get("status") == "pass", registry.get("status"))
    add(checks, "registry_has_model", len(registry.get("models", [])) >= 1, registry.get("models"))

    if registry.get("models"):
        add(
            checks,
            "registry_model_temporal_holdout",
            registry["models"][0].get("temporal_holdout_used") is True,
            registry["models"][0]
        )

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_ranking_baseline_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates Group 3 ranking baseline, temporal holdout enforcement, ranked list outputs, offline evaluation log, and model registry."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "ranking_baseline_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_ranking_baseline_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
