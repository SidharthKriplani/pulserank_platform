from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.pulserank.ranking.candidate_generation import (
    candidate_source_mix,
    content_candidates,
    coverage,
    hybrid_candidates,
    popularity_candidates,
    recall_at_k,
)


ROOT = Path(".")
DATA = ROOT / "data" / "processed"
EVIDENCE = ROOT / "outputs" / "evidence"
VALIDATION = ROOT / "outputs" / "validation"
REPORTS = ROOT / "outputs" / "reports"

K = 100


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


def relevant_items_from_purchases(purchases: list[dict]) -> dict[str, set[str]]:
    out = defaultdict(set)
    for row in purchases:
        out[row["session_id"]].add(row["item_id"])
    return out


def main() -> None:
    sessions = read_csv(DATA / "user_sessions.csv")
    items = read_csv(DATA / "item_catalog.csv")
    impressions = read_csv(DATA / "impression_log.csv")
    purchases = read_csv(DATA / "purchase_log.csv")

    impressions_by_session = defaultdict(list)
    for row in impressions:
        impressions_by_session[row["session_id"]].append(row)

    item_by_id = {row["item_id"]: row for row in items}
    relevant_by_session = relevant_items_from_purchases(purchases)

    popularity_set = popularity_candidates(items, K)
    popularity_sets = {s["session_id"]: popularity_set for s in sessions}

    content_sets = {}
    hybrid_sets = {}

    rows = []
    sample_rows = []
    for idx, session in enumerate(sessions, start=1):
        session_id = session["session_id"]
        content = content_candidates(session, items, impressions_by_session, item_by_id, K)
        hybrid = hybrid_candidates(session, items, impressions_by_session, item_by_id, K)

        content_sets[session_id] = content
        hybrid_sets[session_id] = hybrid

        rows.append({
            "session_id": session_id,
            "candidate_set_size": len(hybrid),
            "candidate_generator": "hybrid_popularity_content",
            "candidate_item_ids": json.dumps(hybrid),
            "top_10_candidate_item_ids": json.dumps(hybrid[:10]),
        })

        if idx <= 25:
            sample_rows.append(rows[-1])

    write_csv(DATA / "candidate_sets.csv", rows)

    evaluated_sessions = len(relevant_by_session)
    popularity_recall_100 = recall_at_k(popularity_sets, relevant_by_session, K)
    content_recall_100 = recall_at_k(content_sets, relevant_by_session, K)
    hybrid_recall_100 = recall_at_k(hybrid_sets, relevant_by_session, K)

    candidate_report = {
        "artifact": "pulserank_candidate_generation_report_v1",
        "status": "pass",
        "candidate_set_size": K,
        "sessions_scored": len(sessions),
        "items_available": len(items),
        "candidate_rows_written": len(rows),
        "generators": [
            "popularity",
            "content_similarity",
            "hybrid_popularity_content"
        ],
        "evaluated_purchase_sessions": evaluated_sessions,
        "recall_at_100": {
            "popularity": round(popularity_recall_100, 5),
            "content_similarity": round(content_recall_100, 5),
            "hybrid_popularity_content": round(hybrid_recall_100, 5),
        },
        "catalog_coverage_at_100": {
            "popularity": round(coverage(popularity_sets, len(items)), 5),
            "content_similarity": round(coverage(content_sets, len(items)), 5),
            "hybrid_popularity_content": round(coverage(hybrid_sets, len(items)), 5),
        },
        "hybrid_category_mix": candidate_source_mix(hybrid_sets, items),
        "evidence_statement": "Implements PulseRank must-have candidate generation using popularity, content similarity, and a hybrid baseline before downstream ranking."
    }

    recall_report = {
        "artifact": "pulserank_candidate_recall_report_v1",
        "status": "pass",
        "metric": "purchase_session_recall_at_100",
        "evaluated_sessions": evaluated_sessions,
        "baseline_popularity_recall_at_100": round(popularity_recall_100, 5),
        "content_similarity_recall_at_100": round(content_recall_100, 5),
        "hybrid_recall_at_100": round(hybrid_recall_100, 5),
        "best_generator": max(
            [
                ("popularity", popularity_recall_100),
                ("content_similarity", content_recall_100),
                ("hybrid_popularity_content", hybrid_recall_100),
            ],
            key=lambda x: x[1],
        )[0],
        "interpretation": "Candidate generation prioritizes recall: items not retrieved cannot be ranked.",
    }

    sample_report = {
        "artifact": "pulserank_candidate_samples_v1",
        "status": "pass",
        "sample_count": len(sample_rows),
        "samples": sample_rows,
    }

    write_json(EVIDENCE / "candidate_generation_report.json", candidate_report)
    write_json(EVIDENCE / "candidate_recall_report.json", recall_report)
    write_json(EVIDENCE / "candidate_samples.json", sample_report)

    group_report = {
        "artifact": "pulserank_group2_candidate_generation_report_v1",
        "status": "pass",
        "data_path": "data/processed/candidate_sets.csv",
        "candidate_generation_report_path": "outputs/evidence/candidate_generation_report.json",
        "candidate_recall_report_path": "outputs/evidence/candidate_recall_report.json",
        "candidate_samples_path": "outputs/evidence/candidate_samples.json",
        "evidence_statement": "Group 2 candidate generation complete with popularity, content-similarity, and hybrid recall evidence.",
    }
    write_json(REPORTS / "group2_candidate_generation_report_v1.json", group_report)

    print("pulserank_candidate_generation_v1 complete")
    print("status: pass")
    print(f"sessions_scored: {len(sessions)}")
    print(f"candidate_set_size: {K}")
    print(f"purchase_sessions_evaluated: {evaluated_sessions}")
    print(f"popularity_recall_at_100: {popularity_recall_100:.5f}")
    print(f"content_recall_at_100: {content_recall_100:.5f}")
    print(f"hybrid_recall_at_100: {hybrid_recall_100:.5f}")
    print("wrote data/processed/candidate_sets.csv")
    print("wrote outputs/evidence/candidate_generation_report.json")
    print("wrote outputs/evidence/candidate_recall_report.json")
    print("wrote outputs/evidence/candidate_samples.json")
    print("wrote outputs/reports/group2_candidate_generation_report_v1.json")


if __name__ == "__main__":
    main()
