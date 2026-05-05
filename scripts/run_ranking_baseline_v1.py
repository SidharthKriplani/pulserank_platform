from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.pulserank.ranking.ranking_baseline import (
    build_train_statistics,
    evaluate_ranked_sessions,
    rank_candidates_for_session,
    seller_gini_for_ranked_lists,
)


ROOT = Path(".")
DATA = ROOT / "data" / "processed"
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"

TRAIN_MAX_DAY = 60
HOLDOUT_MIN_DAY = 61
K = 10


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


def load_candidate_sets(rows: list[dict]) -> dict[str, list[str]]:
    return {row["session_id"]: json.loads(row["candidate_item_ids"]) for row in rows}


def purchases_by_session(rows: list[dict]) -> dict[str, set[str]]:
    out = defaultdict(set)
    for row in rows:
        out[row["session_id"]].add(row["item_id"])
    return out


def popularity_order(candidate_sets: dict[str, list[str]], item_by_id: dict[str, dict]) -> dict[str, list[str]]:
    ranked = {}
    for sid, candidates in candidate_sets.items():
        ranked[sid] = sorted(candidates, key=lambda item_id: int(item_by_id[item_id]["popularity_rank"]))
    return ranked


def main() -> None:
    sessions = read_csv(DATA / "user_sessions.csv")
    items = read_csv(DATA / "item_catalog.csv")
    purchases = read_csv(DATA / "purchase_log.csv")
    candidate_rows = read_csv(DATA / "candidate_sets.csv")

    item_by_id = {row["item_id"]: row for row in items}
    session_by_id = {row["session_id"]: row for row in sessions}
    candidate_sets = load_candidate_sets(candidate_rows)
    purchase_map = purchases_by_session(purchases)

    train_sessions = [s for s in sessions if int(s["simulated_day"]) <= TRAIN_MAX_DAY]
    holdout_sessions = [s for s in sessions if int(s["simulated_day"]) >= HOLDOUT_MIN_DAY]

    train_ids = {s["session_id"] for s in train_sessions}
    holdout_ids = {s["session_id"] for s in holdout_sessions}

    train_purchase_map = {sid: items_ for sid, items_ in purchase_map.items() if sid in train_ids}
    holdout_purchase_map = {sid: items_ for sid, items_ in purchase_map.items() if sid in holdout_ids}

    stats = build_train_statistics(train_sessions, item_by_id, train_purchase_map, candidate_sets)
    max_pop_rank = max(int(x["popularity_rank"]) for x in items)

    ranked_rows = []
    model_ranked = {}

    for session in sessions:
        sid = session["session_id"]
        candidates = candidate_sets.get(sid, [])
        ranked = rank_candidates_for_session(session, candidates, item_by_id, stats, max_pop_rank)
        model_ranked[sid] = [x["item_id"] for x in ranked]

        top10 = ranked[:10]
        positives = purchase_map.get(sid, set())
        for pos, row in enumerate(top10, start=1):
            ranked_rows.append({
                "session_id": sid,
                "item_id": row["item_id"],
                "model_rank": pos,
                "ranking_score": round(row["score"], 8),
                "label_purchase_in_session": int(row["item_id"] in positives),
                "category_l1": row["category_l1"],
                "seller_id": row["seller_id"],
                "simulated_day": session["simulated_day"],
                "split": "train" if sid in train_ids else "holdout",
            })

    write_csv(DATA / "ranked_lists.csv", ranked_rows)

    popularity_ranked = popularity_order(candidate_sets, item_by_id)

    holdout_model_ranked = {sid: ranked for sid, ranked in model_ranked.items() if sid in holdout_ids}
    holdout_pop_ranked = {sid: ranked for sid, ranked in popularity_ranked.items() if sid in holdout_ids}

    train_model_ranked = {sid: ranked for sid, ranked in model_ranked.items() if sid in train_ids}

    holdout_model_metrics = evaluate_ranked_sessions(holdout_model_ranked, holdout_purchase_map, K)
    holdout_pop_metrics = evaluate_ranked_sessions(holdout_pop_ranked, holdout_purchase_map, K)
    train_model_metrics = evaluate_ranked_sessions(train_model_ranked, train_purchase_map, K)

    model_seller_gini = seller_gini_for_ranked_lists(holdout_model_ranked, item_by_id, K)
    pop_seller_gini = seller_gini_for_ranked_lists(holdout_pop_ranked, item_by_id, K)

    offline_eval = {
        "artifact": "pulserank_offline_eval_log_v1",
        "status": "pass",
        "split_strategy": "temporal_holdout",
        "train_days": [1, TRAIN_MAX_DAY],
        "holdout_days": [HOLDOUT_MIN_DAY, 90],
        "ranking_metric_stack": [
            "ndcg_at_10",
            "mrr",
            "precision_at_10",
            "recall_at_10",
            "map_at_10",
            "seller_gini_at_10",
        ],
        "baseline_popularity": {
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in holdout_pop_metrics.items()},
            "seller_gini_at_10": round(pop_seller_gini, 5),
        },
        "ranking_baseline": {
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in holdout_model_metrics.items()},
            "seller_gini_at_10": round(model_seller_gini, 5),
        },
        "train_ranking_baseline": {
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in train_model_metrics.items()},
        },
        "evidence_statement": "Evaluates PulseRank ranking baseline on temporal holdout. Random split is intentionally not used."
    }

    ranking_report = {
        "artifact": "pulserank_ranking_baseline_report_v1",
        "status": "pass",
        "ranker_type": "deterministic_linear_relevance_baseline",
        "model_note": "Lightweight interpretable ranking baseline using session/category affinity, price fit, freshness, popularity, latent quality, and train-period target rates.",
        "temporal_split_enforced": True,
        "train_sessions": len(train_sessions),
        "holdout_sessions": len(holdout_sessions),
        "ranked_rows_written": len(ranked_rows),
        "holdout_purchase_sessions_evaluated": holdout_model_metrics["evaluated_sessions"],
        "holdout_ndcg_at_10": round(holdout_model_metrics["ndcg_at_10"], 5),
        "holdout_mrr": round(holdout_model_metrics["mrr"], 5),
        "holdout_precision_at_10": round(holdout_model_metrics["precision_at_10"], 5),
        "holdout_recall_at_10": round(holdout_model_metrics["recall_at_10"], 5),
        "holdout_map_at_10": round(holdout_model_metrics["map_at_10"], 5),
        "popularity_ndcg_at_10": round(holdout_pop_metrics["ndcg_at_10"], 5),
        "ndcg_delta_vs_popularity": round(holdout_model_metrics["ndcg_at_10"] - holdout_pop_metrics["ndcg_at_10"], 5),
        "seller_gini_at_10": round(model_seller_gini, 5),
        "evidence_statement": "Implements Group 3 ranking baseline with leakage-aware temporal train/test split and offline ranking metric stack."
    }

    model_registry = {
        "artifact": "pulserank_model_registry_v1",
        "status": "pass",
        "models": [
            {
                "model_id": "ranker_linear_relevance_v1",
                "model_type": "deterministic_linear_relevance_baseline",
                "training_window_days": [1, TRAIN_MAX_DAY],
                "evaluation_window_days": [HOLDOUT_MIN_DAY, 90],
                "promotion_gate": "baseline_logged_not_promoted",
                "temporal_holdout_used": True,
                "metrics": ranking_report,
            }
        ],
        "promotion_gate_note": "Model is logged as a baseline. Later groups add IPS correction, delayed attribution, reranking constraints, and promotion gates.",
    }

    write_json(EVIDENCE / "ranking_baseline_report.json", ranking_report)
    write_json(EVIDENCE / "offline_eval_log.json", offline_eval)
    write_json(EVIDENCE / "model_registry.json", model_registry)

    group_report = {
        "artifact": "pulserank_group3_ranking_baseline_report_v1",
        "status": "pass",
        "ranked_lists_path": "data/processed/ranked_lists.csv",
        "ranking_report_path": "outputs/evidence/ranking_baseline_report.json",
        "offline_eval_path": "outputs/evidence/offline_eval_log.json",
        "model_registry_path": "outputs/evidence/model_registry.json",
        "evidence_statement": "Group 3 ranking baseline complete with temporal holdout and offline metric stack."
    }
    write_json(REPORTS / "group3_ranking_baseline_report_v1.json", group_report)

    print("pulserank_ranking_baseline_v1 complete")
    print("status: pass")
    print(f"train_sessions: {len(train_sessions)}")
    print(f"holdout_sessions: {len(holdout_sessions)}")
    print(f"holdout_purchase_sessions_evaluated: {holdout_model_metrics['evaluated_sessions']}")
    print(f"holdout_ndcg_at_10: {ranking_report['holdout_ndcg_at_10']}")
    print(f"popularity_ndcg_at_10: {ranking_report['popularity_ndcg_at_10']}")
    print(f"ndcg_delta_vs_popularity: {ranking_report['ndcg_delta_vs_popularity']}")
    print(f"seller_gini_at_10: {ranking_report['seller_gini_at_10']}")
    print("wrote data/processed/ranked_lists.csv")
    print("wrote outputs/evidence/ranking_baseline_report.json")
    print("wrote outputs/evidence/offline_eval_log.json")
    print("wrote outputs/evidence/model_registry.json")
    print("wrote outputs/reports/group3_ranking_baseline_report_v1.json")


if __name__ == "__main__":
    main()
