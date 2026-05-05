from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any


def parse_json_dict(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


def parse_json_list(value: str) -> list[float]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return []


def price_bucket(price: float) -> str:
    if price <= 900:
        return "budget"
    if price <= 3000:
        return "mid"
    return "premium"


def price_fit(price: float, user_bucket: str) -> float:
    item_bucket = price_bucket(price)
    if user_bucket == item_bucket:
        return 1.0
    if user_bucket == "mid":
        return 0.65
    if user_bucket == "budget" and item_bucket == "mid":
        return 0.55
    if user_bucket == "premium" and item_bucket == "mid":
        return 0.70
    return 0.30


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def safe_rate(pos: int, total: int, prior_pos: int = 3, prior_total: int = 100) -> float:
    return (pos + prior_pos) / (total + prior_total)


def build_train_statistics(
    train_sessions: list[dict[str, Any]],
    item_by_id: dict[str, dict[str, Any]],
    purchases_by_session: dict[str, set[str]],
    candidate_sets: dict[str, list[str]],
) -> dict[str, Any]:
    category_total = Counter()
    category_pos = Counter()
    seller_total = Counter()
    seller_pos = Counter()
    price_total = Counter()
    price_pos = Counter()

    for session in train_sessions:
        sid = session["session_id"]
        positives = purchases_by_session.get(sid, set())
        for item_id in candidate_sets.get(sid, [])[:100]:
            item = item_by_id[item_id]
            label = int(item_id in positives)
            category = item["category_l1"]
            seller = item["seller_id"]
            pb = price_bucket(float(item["price"]))

            category_total[category] += 1
            seller_total[seller] += 1
            price_total[pb] += 1
            if label:
                category_pos[category] += 1
                seller_pos[seller] += 1
                price_pos[pb] += 1

    return {
        "category_rate": {k: safe_rate(category_pos[k], category_total[k]) for k in category_total},
        "seller_rate": {k: safe_rate(seller_pos[k], seller_total[k]) for k in seller_total},
        "price_bucket_rate": {k: safe_rate(price_pos[k], price_total[k]) for k in price_total},
        "global_rate": safe_rate(sum(category_pos.values()), sum(category_total.values())),
        "category_total": dict(category_total),
        "seller_total": dict(seller_total),
    }


def feature_vector(
    session: dict[str, Any],
    item: dict[str, Any],
    stats: dict[str, Any],
    max_popularity_rank: int,
) -> dict[str, float]:
    affinity = parse_json_dict(session.get("category_affinity", "{}"))
    category = item["category_l1"]
    seller = item["seller_id"]
    price = float(item["price"])
    user_bucket = session.get("price_sensitivity_bucket", "mid")
    pop_rank = int(item["popularity_rank"])

    return {
        "category_affinity": affinity.get(category, 0.05),
        "price_fit": price_fit(price, user_bucket),
        "freshness_score": float(item.get("freshness_score", 0.0)),
        "latent_quality": float(item.get("latent_quality", 0.0)),
        "popularity_score": 1.0 - ((pop_rank - 1) / max(1, max_popularity_rank)),
        "category_train_rate": stats["category_rate"].get(category, stats["global_rate"]),
        "seller_train_rate": stats["seller_rate"].get(seller, stats["global_rate"]),
        "price_bucket_train_rate": stats["price_bucket_rate"].get(price_bucket(price), stats["global_rate"]),
    }


def score_candidate(features: dict[str, float]) -> float:
    linear = (
        -4.25
        + 1.65 * features["category_affinity"]
        + 0.78 * features["price_fit"]
        + 0.70 * features["freshness_score"]
        + 1.35 * features["latent_quality"]
        + 0.92 * features["popularity_score"]
        + 7.50 * features["category_train_rate"]
        + 5.20 * features["seller_train_rate"]
        + 4.10 * features["price_bucket_train_rate"]
    )
    return sigmoid(linear)


def rank_candidates_for_session(
    session: dict[str, Any],
    candidate_item_ids: list[str],
    item_by_id: dict[str, dict[str, Any]],
    stats: dict[str, Any],
    max_popularity_rank: int,
) -> list[dict[str, Any]]:
    scored = []
    for item_id in candidate_item_ids:
        item = item_by_id[item_id]
        features = feature_vector(session, item, stats, max_popularity_rank)
        score = score_candidate(features)
        scored.append({
            "item_id": item_id,
            "score": score,
            "features": features,
            "category_l1": item["category_l1"],
            "seller_id": item["seller_id"],
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def dcg_at_k(labels: list[int], k: int) -> float:
    total = 0.0
    for idx, label in enumerate(labels[:k], start=1):
        gain = (2 ** label) - 1
        total += gain / math.log2(idx + 1)
    return total


def ndcg_at_k(labels: list[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(labels, k) / denom


def reciprocal_rank(labels: list[int]) -> float:
    for idx, label in enumerate(labels, start=1):
        if label:
            return 1 / idx
    return 0.0


def precision_at_k(labels: list[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(labels[:k]) / k


def recall_at_k(labels: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(labels[:k]) / total_relevant


def average_precision_at_k(labels: list[int], k: int) -> float:
    hits = 0
    precisions = []
    for idx, label in enumerate(labels[:k], start=1):
        if label:
            hits += 1
            precisions.append(hits / idx)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def evaluate_ranked_sessions(
    ranked_by_session: dict[str, list[str]],
    purchases_by_session: dict[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    rows = []
    for session_id, positives in purchases_by_session.items():
        if not positives:
            continue
        ranked = ranked_by_session.get(session_id, [])
        labels = [1 if item_id in positives else 0 for item_id in ranked]
        if not labels:
            continue
        rows.append({
            "ndcg": ndcg_at_k(labels, k),
            "mrr": reciprocal_rank(labels),
            "precision": precision_at_k(labels, k),
            "recall": recall_at_k(labels, len(positives), k),
            "map": average_precision_at_k(labels, k),
        })

    if not rows:
        return {
            "evaluated_sessions": 0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
            "precision_at_10": 0.0,
            "recall_at_10": 0.0,
            "map_at_10": 0.0,
        }

    return {
        "evaluated_sessions": len(rows),
        "ndcg_at_10": sum(x["ndcg"] for x in rows) / len(rows),
        "mrr": sum(x["mrr"] for x in rows) / len(rows),
        "precision_at_10": sum(x["precision"] for x in rows) / len(rows),
        "recall_at_10": sum(x["recall"] for x in rows) / len(rows),
        "map_at_10": sum(x["map"] for x in rows) / len(rows),
    }


def seller_gini_for_ranked_lists(
    ranked_by_session: dict[str, list[str]],
    item_by_id: dict[str, dict[str, Any]],
    k: int = 10,
) -> float:
    exposure = Counter()
    for ranked in ranked_by_session.values():
        for item_id in ranked[:k]:
            seller = item_by_id[item_id]["seller_id"]
            exposure[seller] += 1

    values = sorted(exposure.values())
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0

    cumulative = 0
    for idx, val in enumerate(values, start=1):
        cumulative += idx * val
    return (2 * cumulative) / (n * total) - (n + 1) / n
