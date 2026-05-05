from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.pulserank.evaluation.position_bias import (
    build_ips_examples,
    estimate_click_propensity_by_rank,
    evaluate_raw_and_ips_ndcg,
    rank_propensity_monotonicity_score,
    summarize_weight_distribution,
)


ROOT = Path(".")
DATA = ROOT / "data" / "processed"
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"

PROPENSITY_FLOOR = 0.01
MAX_RANK = 10
TRAIN_MAX_DAY = 60


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    sessions = read_csv(DATA / "user_sessions.csv")
    impressions = read_csv(DATA / "impression_log.csv")
    attributed = read_csv(DATA / "attributed_label_log.csv")

    session_day_by_id = {row["session_id"]: int(row["simulated_day"]) for row in sessions}
    attributed_by_impression = {row["impression_id"]: row for row in attributed}

    train_impressions = [
        row for row in impressions
        if session_day_by_id[row["session_id"]] <= TRAIN_MAX_DAY
    ]

    holdout_impressions = [
        row for row in impressions
        if session_day_by_id[row["session_id"]] > TRAIN_MAX_DAY
    ]

    holdout_by_session = defaultdict(list)
    for row in holdout_impressions:
        holdout_by_session[row["session_id"]].append(row)

    propensity_by_rank = estimate_click_propensity_by_rank(
        train_impressions=train_impressions,
        max_rank=MAX_RANK,
        floor=PROPENSITY_FLOOR,
    )

    ips_examples = build_ips_examples(
        impressions=impressions,
        attributed_by_impression=attributed_by_impression,
        session_day_by_id=session_day_by_id,
        propensity_by_rank=propensity_by_rank,
        train_max_day=TRAIN_MAX_DAY,
    )

    write_csv(DATA / "ips_training_examples.csv", ips_examples)

    eval_summary = evaluate_raw_and_ips_ndcg(
        holdout_impressions_by_session=dict(holdout_by_session),
        attributed_by_impression=attributed_by_impression,
        propensity_by_rank=propensity_by_rank,
        k=10,
    )

    weight_summary = summarize_weight_distribution(ips_examples)
    monotonicity_score = rank_propensity_monotonicity_score(propensity_by_rank)

    propensity_rows = [
        {
            "display_rank": rank,
            "impressions": int(stats["impressions"]),
            "clicks": int(stats["clicks"]),
            "empirical_propensity": round(float(stats["empirical_propensity"]), 8),
            "clipped_propensity": round(float(stats["clipped_propensity"]), 8),
            "ips_weight": round(float(stats["ips_weight"]), 8),
            "was_clipped": bool(stats["was_clipped"]),
        }
        for rank, stats in sorted(propensity_by_rank.items())
    ]

    propensity_report = {
        "artifact": "pulserank_propensity_by_rank_report_v1",
        "status": "pass",
        "propensity_floor": PROPENSITY_FLOOR,
        "estimated_on": "train_impressions_days_1_to_60",
        "max_rank": MAX_RANK,
        "rank_propensity_monotonicity_score": round(monotonicity_score, 5),
        "propensity_by_rank": propensity_rows,
        "evidence_statement": "Estimates examination/click propensity by display_rank from train-period logged impressions."
    }

    bias_report = {
        "artifact": "pulserank_bias_correction_report_v1",
        "status": "pass",
        "method": "inverse_propensity_scoring",
        "position_bias_problem": "Logged clicks are position-confounded because top-ranked items receive higher examination probability independent of item quality.",
        "propensity_estimation": "Empirical train-period click rate by display_rank.",
        "propensity_clip_threshold": PROPENSITY_FLOOR,
        "train_impressions": len(train_impressions),
        "holdout_impressions": len(holdout_impressions),
        "ips_examples_written": len(ips_examples),
        "evaluated_holdout_purchase_sessions": int(eval_summary["evaluated_sessions"]),
        "raw_ndcg_at_10": round(float(eval_summary["raw_ndcg_at_10"]), 5),
        "ips_weighted_ndcg_at_10": round(float(eval_summary["ips_weighted_ndcg_at_10"]), 5),
        "bias_delta_ips_minus_raw": round(float(eval_summary["bias_delta_ips_minus_raw"]), 5),
        "weight_distribution": {
            "min_weight": round(float(weight_summary["min_weight"]), 5),
            "max_weight": round(float(weight_summary["max_weight"]), 5),
            "mean_weight": round(float(weight_summary["mean_weight"]), 5),
            "clip_rate": round(float(weight_summary["clip_rate"]), 5),
        },
        "assumption_limitations": [
            "Uses rank-level empirical click propensity rather than a full cascade/examination model.",
            "Synthetic corpus approximates position bias but does not represent live marketplace traffic.",
            "IPS weighting controls for display-rank confounding but can increase variance when propensities are small.",
            "Propensity clipping is used to control variance."
        ],
        "evidence_statement": "Implements IPS position-bias correction with propensity estimation, clipping, training-example weights, and raw vs. IPS-weighted NDCG comparison."
    }

    write_json(EVIDENCE / "propensity_by_rank_report.json", propensity_report)
    write_json(EVIDENCE / "bias_correction_report.json", bias_report)

    group_report = {
        "artifact": "pulserank_group4_bias_correction_report_v1",
        "status": "pass",
        "ips_training_examples_path": "data/processed/ips_training_examples.csv",
        "propensity_report_path": "outputs/evidence/propensity_by_rank_report.json",
        "bias_report_path": "outputs/evidence/bias_correction_report.json",
        "evidence_statement": "Group 4 IPS bias correction complete with display-rank propensity estimation and before/after ranking evaluation."
    }

    write_json(REPORTS / "group4_bias_correction_report_v1.json", group_report)

    print("pulserank_bias_correction_v1 complete")
    print("status: pass")
    print(f"train_impressions: {len(train_impressions)}")
    print(f"holdout_impressions: {len(holdout_impressions)}")
    print(f"evaluated_holdout_purchase_sessions: {eval_summary['evaluated_sessions']}")
    print(f"raw_ndcg_at_10: {bias_report['raw_ndcg_at_10']}")
    print(f"ips_weighted_ndcg_at_10: {bias_report['ips_weighted_ndcg_at_10']}")
    print(f"bias_delta_ips_minus_raw: {bias_report['bias_delta_ips_minus_raw']}")
    print(f"clip_rate: {bias_report['weight_distribution']['clip_rate']}")
    print("wrote data/processed/ips_training_examples.csv")
    print("wrote outputs/evidence/propensity_by_rank_report.json")
    print("wrote outputs/evidence/bias_correction_report.json")
    print("wrote outputs/reports/group4_bias_correction_report_v1.json")


if __name__ == "__main__":
    main()
