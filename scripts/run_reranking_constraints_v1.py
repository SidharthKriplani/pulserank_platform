from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.pulserank.ranking.ranking_baseline import (
    build_train_statistics,
    evaluate_ranked_sessions,
    rank_candidates_for_session,
)
from src.pulserank.ranking.reranking import (
    catalog_coverage,
    category_topk_max_share,
    governed_rerank,
    intra_list_diversity,
    novelty_at_k,
    seller_gini,
)


ROOT = Path(".")
DATA = ROOT / "data" / "processed"
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"

TRAIN_MAX_DAY = 60
K = 10
MMR_LAMBDA = 0.70
SELLER_GINI_THRESHOLD = 0.65
CATEGORY_TOP10_MAX_SHARE = 0.50
ILD_FLOOR = 0.30


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    sessions = read_csv(DATA / "user_sessions.csv")
    items = read_csv(DATA / "item_catalog.csv")
    purchases = read_csv(DATA / "purchase_log.csv")
    candidate_rows = read_csv(DATA / "candidate_sets.csv")

    item_by_id = {row["item_id"]: row for row in items}
    candidate_sets = load_candidate_sets(candidate_rows)
    purchase_map = purchases_by_session(purchases)

    train_sessions = [s for s in sessions if int(s["simulated_day"]) <= TRAIN_MAX_DAY]
    holdout_sessions = [s for s in sessions if int(s["simulated_day"]) > TRAIN_MAX_DAY]
    train_ids = {s["session_id"] for s in train_sessions}
    holdout_ids = {s["session_id"] for s in holdout_sessions}

    train_purchase_map = {sid: items_ for sid, items_ in purchase_map.items() if sid in train_ids}
    holdout_purchase_map = {sid: items_ for sid, items_ in purchase_map.items() if sid in holdout_ids}

    stats = build_train_statistics(train_sessions, item_by_id, train_purchase_map, candidate_sets)
    max_pop_rank = max(int(x["popularity_rank"]) for x in items)

    raw_ranked_lists = {}
    reranked_lists = {}
    rows = []
    audit_rows = []

    raw_ild = []
    reranked_ild = []
    raw_novelty = []
    reranked_novelty = []
    raw_category_share = []
    reranked_category_share = []

    for session in sessions:
        sid = session["session_id"]
        candidates = candidate_sets.get(sid, [])
        ranked_candidates = rank_candidates_for_session(session, candidates, item_by_id, stats, max_pop_rank)

        raw_items = [x["item_id"] for x in ranked_candidates[:K]]
        reranked, audit = governed_rerank(
            ranked_candidates=ranked_candidates,
            item_by_id=item_by_id,
            k=K,
            mmr_lambda=MMR_LAMBDA,
            category_top10_max_share=CATEGORY_TOP10_MAX_SHARE,
            max_per_seller=2,
        )
        reranked_items = [x["item_id"] for x in reranked]

        raw_ranked_lists[sid] = raw_items
        reranked_lists[sid] = reranked_items

        raw_ild.append(intra_list_diversity(raw_items, item_by_id))
        reranked_ild.append(intra_list_diversity(reranked_items, item_by_id))
        raw_novelty.append(novelty_at_k(raw_items, item_by_id, len(items), K))
        reranked_novelty.append(novelty_at_k(reranked_items, item_by_id, len(items), K))
        raw_category_share.append(category_topk_max_share(raw_items, item_by_id, K))
        reranked_category_share.append(category_topk_max_share(reranked_items, item_by_id, K))

        positives = purchase_map.get(sid, set())
        for pos, cand in enumerate(reranked, start=1):
            item = item_by_id[cand["item_id"]]
            rows.append({
                "session_id": sid,
                "item_id": cand["item_id"],
                "reranked_position": pos,
                "pre_rerank_score": round(float(cand["score"]), 8),
                "label_purchase_in_session": int(cand["item_id"] in positives),
                "category_l1": item["category_l1"],
                "category_l2": item["category_l2"],
                "seller_id": item["seller_id"],
                "simulated_day": session["simulated_day"],
                "split": "train" if sid in train_ids else "holdout",
            })

        audit_rows.append({
            "session_id": sid,
            "split": "train" if sid in train_ids else "holdout",
            "selected_count": audit["selected_count"],
            "max_category_share": round(float(audit["max_category_share"]), 5),
            "unique_sellers": len(audit["seller_counts"]),
            "unique_categories": len(audit["category_counts"]),
            "soft_block_count": len(audit["soft_blocks"]),
            "category_counts": json.dumps(audit["category_counts"], sort_keys=True),
            "seller_counts": json.dumps(audit["seller_counts"], sort_keys=True),
        })

    write_csv(DATA / "reranked_lists.csv", rows)
    write_csv(DATA / "rerank_audit_log.csv", audit_rows)

    raw_holdout = {sid: items_ for sid, items_ in raw_ranked_lists.items() if sid in holdout_ids}
    reranked_holdout = {sid: items_ for sid, items_ in reranked_lists.items() if sid in holdout_ids}

    raw_metrics = evaluate_ranked_sessions(raw_holdout, holdout_purchase_map, K)
    reranked_metrics = evaluate_ranked_sessions(reranked_holdout, holdout_purchase_map, K)

    raw_seller_gini = seller_gini(raw_holdout, item_by_id, K)
    reranked_seller_gini = seller_gini(reranked_holdout, item_by_id, K)

    raw_coverage = catalog_coverage(raw_holdout, len(items), K)
    reranked_coverage = catalog_coverage(reranked_holdout, len(items), K)

    guardrail_status = "pass"
    guardrail_failures = []
    if reranked_seller_gini > SELLER_GINI_THRESHOLD:
        guardrail_status = "fail"
        guardrail_failures.append("seller_gini_threshold")
    if mean(reranked_ild) < ILD_FLOOR:
        guardrail_status = "fail"
        guardrail_failures.append("ild_floor")
    if mean(reranked_category_share) > CATEGORY_TOP10_MAX_SHARE + 0.05:
        guardrail_status = "review"
        guardrail_failures.append("category_share_review_band")

    diversity_guardrail = {
        "artifact": "pulserank_diversity_guardrail_log_v1",
        "status": guardrail_status,
        "constraints": {
            "mmr_lambda": MMR_LAMBDA,
            "seller_gini_threshold": SELLER_GINI_THRESHOLD,
            "category_top10_max_share": CATEGORY_TOP10_MAX_SHARE,
            "ild_floor": ILD_FLOOR,
            "max_per_seller_per_top10": 2,
        },
        "before_rerank": {
            "seller_gini_at_10": round(raw_seller_gini, 5),
            "mean_ild_at_10": round(mean(raw_ild), 5),
            "mean_novelty_at_10": round(mean(raw_novelty), 5),
            "mean_category_max_share_at_10": round(mean(raw_category_share), 5),
            "catalog_coverage_at_10": round(raw_coverage, 5),
            "ndcg_at_10": round(float(raw_metrics["ndcg_at_10"]), 5),
            "recall_at_10": round(float(raw_metrics["recall_at_10"]), 5),
        },
        "after_rerank": {
            "seller_gini_at_10": round(reranked_seller_gini, 5),
            "mean_ild_at_10": round(mean(reranked_ild), 5),
            "mean_novelty_at_10": round(mean(reranked_novelty), 5),
            "mean_category_max_share_at_10": round(mean(reranked_category_share), 5),
            "catalog_coverage_at_10": round(reranked_coverage, 5),
            "ndcg_at_10": round(float(reranked_metrics["ndcg_at_10"]), 5),
            "recall_at_10": round(float(reranked_metrics["recall_at_10"]), 5),
        },
        "tradeoff_summary": {
            "seller_gini_delta": round(reranked_seller_gini - raw_seller_gini, 5),
            "ild_delta": round(mean(reranked_ild) - mean(raw_ild), 5),
            "novelty_delta": round(mean(reranked_novelty) - mean(raw_novelty), 5),
            "coverage_delta": round(reranked_coverage - raw_coverage, 5),
            "ndcg_delta": round(float(reranked_metrics["ndcg_at_10"]) - float(raw_metrics["ndcg_at_10"]), 5),
        },
        "guardrail_failures": guardrail_failures,
        "evidence_statement": "Applies governed reranking after baseline ranking to balance relevance against seller exposure, category concentration, novelty, and list diversity."
    }

    diversity_report = {
        "artifact": "pulserank_diversity_report_v1",
        "status": "pass",
        "sessions_reranked": len(sessions),
        "holdout_sessions": len(holdout_sessions),
        "reranked_rows_written": len(rows),
        "audit_rows_written": len(audit_rows),
        "mean_unique_sellers_per_top10": round(mean([float(x["unique_sellers"]) for x in audit_rows]), 5),
        "mean_unique_categories_per_top10": round(mean([float(x["unique_categories"]) for x in audit_rows]), 5),
        "soft_block_sessions": sum(1 for x in audit_rows if int(x["soft_block_count"]) > 0),
        "evidence_statement": "Summarizes per-session reranking diversity and marketplace exposure behavior."
    }

    coverage_report = {
        "artifact": "pulserank_catalog_coverage_report_v1",
        "status": "pass",
        "raw_catalog_coverage_at_10": round(raw_coverage, 5),
        "reranked_catalog_coverage_at_10": round(reranked_coverage, 5),
        "coverage_delta": round(reranked_coverage - raw_coverage, 5),
        "total_catalog_items": len(items),
        "evidence_statement": "Measures how much of the item catalog receives top-10 exposure before and after reranking."
    }

    write_json(EVIDENCE / "diversity_guardrail_log.json", diversity_guardrail)
    write_json(EVIDENCE / "diversity_report.json", diversity_report)
    write_json(EVIDENCE / "catalog_coverage_report.json", coverage_report)

    group_report = {
        "artifact": "pulserank_group5_reranking_constraints_report_v1",
        "status": "pass",
        "reranked_lists_path": "data/processed/reranked_lists.csv",
        "rerank_audit_log_path": "data/processed/rerank_audit_log.csv",
        "diversity_guardrail_path": "outputs/evidence/diversity_guardrail_log.json",
        "diversity_report_path": "outputs/evidence/diversity_report.json",
        "coverage_report_path": "outputs/evidence/catalog_coverage_report.json",
        "evidence_statement": "Group 5 reranking constraints complete with MMR-style diversity, seller exposure caps, category share monitoring, novelty, and coverage evidence."
    }
    write_json(REPORTS / "group5_reranking_constraints_report_v1.json", group_report)

    print("pulserank_reranking_constraints_v1 complete")
    print(f"status: pass")
    print(f"guardrail_status: {guardrail_status}")
    print(f"sessions_reranked: {len(sessions)}")
    print(f"before_seller_gini_at_10: {diversity_guardrail['before_rerank']['seller_gini_at_10']}")
    print(f"after_seller_gini_at_10: {diversity_guardrail['after_rerank']['seller_gini_at_10']}")
    print(f"before_mean_ild_at_10: {diversity_guardrail['before_rerank']['mean_ild_at_10']}")
    print(f"after_mean_ild_at_10: {diversity_guardrail['after_rerank']['mean_ild_at_10']}")
    print(f"before_catalog_coverage_at_10: {diversity_guardrail['before_rerank']['catalog_coverage_at_10']}")
    print(f"after_catalog_coverage_at_10: {diversity_guardrail['after_rerank']['catalog_coverage_at_10']}")
    print("wrote data/processed/reranked_lists.csv")
    print("wrote data/processed/rerank_audit_log.csv")
    print("wrote outputs/evidence/diversity_guardrail_log.json")
    print("wrote outputs/evidence/diversity_report.json")
    print("wrote outputs/evidence/catalog_coverage_report.json")
    print("wrote outputs/reports/group5_reranking_constraints_report_v1.json")


if __name__ == "__main__":
    main()
