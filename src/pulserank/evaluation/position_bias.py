from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def estimate_click_propensity_by_rank(
    train_impressions: list[dict[str, Any]],
    max_rank: int = 10,
    floor: float = 0.01,
) -> dict[int, dict[str, float]]:
    stats = {}
    for rank in range(1, max_rank + 1):
        rows = [r for r in train_impressions if int(r["display_rank"]) == rank]
        impressions = len(rows)
        clicks = sum(int(r["clicked"]) for r in rows)
        empirical = clicks / impressions if impressions else 0.0
        clipped = max(empirical, floor)
        stats[rank] = {
            "display_rank": rank,
            "impressions": impressions,
            "clicks": clicks,
            "empirical_propensity": empirical,
            "clipped_propensity": clipped,
            "ips_weight": 1.0 / clipped if clipped > 0 else 1.0 / floor,
            "was_clipped": empirical < floor,
        }
    return stats


def dcg(values: list[float], k: int) -> float:
    total = 0.0
    for idx, value in enumerate(values[:k], start=1):
        total += float(value) / math.log2(idx + 1)
    return total


def ndcg(values: list[float], k: int = 10) -> float:
    if not values:
        return 0.0
    denom = dcg(sorted(values, reverse=True), k)
    if denom == 0:
        return 0.0
    return dcg(values, k) / denom


def evaluate_raw_and_ips_ndcg(
    holdout_impressions_by_session: dict[str, list[dict[str, Any]]],
    attributed_by_impression: dict[str, dict[str, Any]],
    propensity_by_rank: dict[int, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    raw_scores = []
    ips_scores = []
    evaluated_sessions = 0

    for session_id, rows in holdout_impressions_by_session.items():
        ordered = sorted(rows, key=lambda r: int(r["display_rank"]))[:k]

        raw_labels = []
        ips_labels = []

        for row in ordered:
            attr = attributed_by_impression.get(row["impression_id"], {})
            label = int(attr.get("attributed_purchase", 0))
            rank = int(row["display_rank"])
            weight = float(propensity_by_rank[rank]["ips_weight"])

            raw_labels.append(float(label))
            ips_labels.append(float(label) * weight)

        if sum(raw_labels) <= 0:
            continue

        evaluated_sessions += 1
        raw_scores.append(ndcg(raw_labels, k))
        ips_scores.append(ndcg(ips_labels, k))

    if evaluated_sessions == 0:
        return {
            "evaluated_sessions": 0,
            "raw_ndcg_at_10": 0.0,
            "ips_weighted_ndcg_at_10": 0.0,
            "bias_delta_ips_minus_raw": 0.0,
        }

    raw_mean = sum(raw_scores) / len(raw_scores)
    ips_mean = sum(ips_scores) / len(ips_scores)

    return {
        "evaluated_sessions": evaluated_sessions,
        "raw_ndcg_at_10": raw_mean,
        "ips_weighted_ndcg_at_10": ips_mean,
        "bias_delta_ips_minus_raw": ips_mean - raw_mean,
    }


def build_ips_examples(
    impressions: list[dict[str, Any]],
    attributed_by_impression: dict[str, dict[str, Any]],
    session_day_by_id: dict[str, int],
    propensity_by_rank: dict[int, dict[str, float]],
    train_max_day: int = 60,
) -> list[dict[str, Any]]:
    rows = []

    for row in impressions:
        rank = int(row["display_rank"])
        prop = propensity_by_rank[rank]
        attr = attributed_by_impression.get(row["impression_id"], {})
        day = int(session_day_by_id[row["session_id"]])
        split = "train" if day <= train_max_day else "holdout"

        rows.append(
            {
                "impression_id": row["impression_id"],
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "display_rank": rank,
                "split": split,
                "clicked": int(row["clicked"]),
                "added_to_cart": int(row["added_to_cart"]),
                "attributed_purchase": int(attr.get("attributed_purchase", 0)),
                "attributed_revenue": float(attr.get("attributed_revenue", 0.0)),
                "empirical_propensity": round(float(prop["empirical_propensity"]), 8),
                "clipped_propensity": round(float(prop["clipped_propensity"]), 8),
                "ips_weight": round(float(prop["ips_weight"]), 8),
                "propensity_was_clipped": bool(prop["was_clipped"]),
            }
        )

    return rows


def summarize_weight_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    weights = [float(r["ips_weight"]) for r in rows]
    if not weights:
        return {
            "min_weight": 0.0,
            "max_weight": 0.0,
            "mean_weight": 0.0,
            "clip_rate": 0.0,
        }

    clip_rate = sum(1 for r in rows if r["propensity_was_clipped"]) / len(rows)

    return {
        "min_weight": min(weights),
        "max_weight": max(weights),
        "mean_weight": sum(weights) / len(weights),
        "clip_rate": clip_rate,
    }


def rank_propensity_monotonicity_score(propensity_by_rank: dict[int, dict[str, float]]) -> float:
    ranks = sorted(propensity_by_rank)
    props = [float(propensity_by_rank[r]["empirical_propensity"]) for r in ranks]
    comparisons = 0
    monotonic_hits = 0

    for i in range(len(props) - 1):
        comparisons += 1
        if props[i] >= props[i + 1]:
            monotonic_hits += 1

    return monotonic_hits / comparisons if comparisons else 0.0
