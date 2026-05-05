from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any


def parse_vec(value: str) -> list[float]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def item_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    if a["category_l1"] != b["category_l1"]:
        category_distance = 1.0
    elif a["category_l2"] != b["category_l2"]:
        category_distance = 0.65
    else:
        category_distance = 0.25

    va = parse_vec(a.get("content_embedding", "[]"))
    vb = parse_vec(b.get("content_embedding", "[]"))
    vector_distance = (1 - cosine_similarity(va, vb)) / 2

    return max(0.0, min(1.0, 0.55 * category_distance + 0.45 * vector_distance))


def intra_list_diversity(item_ids: list[str], item_by_id: dict[str, dict[str, Any]]) -> float:
    if len(item_ids) <= 1:
        return 0.0

    distances = []
    for i in range(len(item_ids)):
        for j in range(i + 1, len(item_ids)):
            distances.append(item_distance(item_by_id[item_ids[i]], item_by_id[item_ids[j]]))

    return sum(distances) / len(distances) if distances else 0.0


def seller_gini(item_lists: dict[str, list[str]], item_by_id: dict[str, dict[str, Any]], k: int = 10) -> float:
    exposure = Counter()

    for item_ids in item_lists.values():
        for item_id in item_ids[:k]:
            exposure[item_by_id[item_id]["seller_id"]] += 1

    values = sorted(exposure.values())
    n = len(values)
    if n == 0:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    cumulative = sum((idx + 1) * val for idx, val in enumerate(values))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def category_topk_max_share(item_ids: list[str], item_by_id: dict[str, dict[str, Any]], k: int = 10) -> float:
    top = item_ids[:k]
    if not top:
        return 0.0
    counts = Counter(item_by_id[item_id]["category_l1"] for item_id in top)
    return max(counts.values()) / len(top)


def catalog_coverage(item_lists: dict[str, list[str]], total_items: int, k: int = 10) -> float:
    exposed = set()
    for item_ids in item_lists.values():
        exposed.update(item_ids[:k])
    return len(exposed) / total_items if total_items else 0.0


def novelty_at_k(item_ids: list[str], item_by_id: dict[str, dict[str, Any]], total_items: int, k: int = 10) -> float:
    top = item_ids[:k]
    if not top:
        return 0.0

    scores = []
    for item_id in top:
        rank = int(item_by_id[item_id]["popularity_rank"])
        scores.append(rank / total_items)

    return sum(scores) / len(scores)


def relevance_ndcg(labels: list[int], k: int = 10) -> float:
    def dcg(vals: list[int]) -> float:
        return sum(((2 ** v) - 1) / math.log2(i + 2) for i, v in enumerate(vals[:k]))

    ideal = sorted(labels, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(labels) / denom


def governed_rerank(
    ranked_candidates: list[dict[str, Any]],
    item_by_id: dict[str, dict[str, Any]],
    k: int = 10,
    mmr_lambda: float = 0.70,
    category_top10_max_share: float = 0.50,
    max_per_seller: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidate_pool = list(ranked_candidates)

    category_counts: Counter[str] = Counter()
    seller_counts: Counter[str] = Counter()
    soft_blocks = []

    while candidate_pool and len(selected) < k:
        best_idx = None
        best_score = -10**9

        for idx, cand in enumerate(candidate_pool):
            item = item_by_id[cand["item_id"]]
            category = item["category_l1"]
            seller = item["seller_id"]

            if seller_counts[seller] >= max_per_seller:
                penalty = 0.35
            else:
                penalty = 0.0

            future_category_share = (category_counts[category] + 1) / (len(selected) + 1)
            if future_category_share > category_top10_max_share:
                penalty += 0.28

            if selected:
                min_distance = min(
                    item_distance(item, item_by_id[s["item_id"]])
                    for s in selected
                )
            else:
                min_distance = 1.0

            novelty = int(item["popularity_rank"]) / max(1, len(item_by_id))
            score = (
                mmr_lambda * float(cand["score"])
                + (1 - mmr_lambda) * min_distance
                + 0.04 * novelty
                - penalty
            )

            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = candidate_pool.pop(best_idx)
        item = item_by_id[chosen["item_id"]]
        category = item["category_l1"]
        seller = item["seller_id"]

        if seller_counts[seller] >= max_per_seller:
            soft_blocks.append({
                "item_id": chosen["item_id"],
                "reason": "seller_cap_exceeded_but_selected_due_to_candidate_shortage",
                "seller_id": seller,
            })

        selected.append(chosen)
        category_counts[category] += 1
        seller_counts[seller] += 1

    if len(selected) < k:
        for cand in candidate_pool:
            if len(selected) >= k:
                break
            selected.append(cand)

    audit = {
        "selected_count": len(selected),
        "category_counts": dict(category_counts),
        "seller_counts": dict(seller_counts),
        "soft_blocks": soft_blocks,
        "max_category_share": max(category_counts.values()) / len(selected) if selected else 0.0,
        "max_per_seller": max_per_seller,
        "mmr_lambda": mmr_lambda,
    }

    return selected, audit
