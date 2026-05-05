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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def price_fit_score(price: float, bucket: str) -> float:
    if bucket == "budget":
        if price <= 900:
            return 1.0
        if price <= 2500:
            return 0.55
        return 0.20
    if bucket == "mid":
        if 700 <= price <= 3500:
            return 1.0
        if price < 700:
            return 0.65
        return 0.45
    if bucket == "premium":
        if price >= 1500:
            return 1.0
        return 0.45
    return 0.50


def popularity_candidates(items: list[dict[str, Any]], k: int) -> list[str]:
    sorted_items = sorted(items, key=lambda x: int(x["popularity_rank"]))
    return [x["item_id"] for x in sorted_items[:k]]


def session_profile_from_history(
    session: dict[str, Any],
    impressions_by_session: dict[str, list[dict[str, Any]]],
    item_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    affinity = parse_json_dict(session.get("category_affinity", "{}"))
    historical = impressions_by_session.get(session["session_id"], [])
    clicked_items = [x for x in historical if str(x.get("clicked", "0")) == "1"]
    source = clicked_items or historical[:3]

    vectors = []
    for imp in source:
        item = item_by_id.get(imp["item_id"])
        if item:
            vectors.append(parse_json_list(item.get("content_embedding", "[]")))

    if vectors:
        dim = len(vectors[0])
        avg_vector = [sum(v[i] for v in vectors if len(v) == dim) / len(vectors) for i in range(dim)]
    else:
        avg_vector = []

    return {
        "category_affinity": affinity,
        "price_sensitivity_bucket": session.get("price_sensitivity_bucket", "mid"),
        "avg_embedding": avg_vector,
    }


def content_similarity_score(profile: dict[str, Any], item: dict[str, Any]) -> float:
    affinity = profile["category_affinity"]
    cat_score = float(affinity.get(item["category_l1"], 0.05))
    price_score = price_fit_score(float(item["price"]), profile["price_sensitivity_bucket"])
    item_vec = parse_json_list(item.get("content_embedding", "[]"))
    vector_score = (cosine_similarity(profile["avg_embedding"], item_vec) + 1) / 2 if profile["avg_embedding"] else 0.5
    freshness = float(item.get("freshness_score", 0.0))
    popularity = 1.0 / max(1, int(item["popularity_rank"]))

    return (
        0.42 * cat_score
        + 0.22 * price_score
        + 0.18 * vector_score
        + 0.10 * freshness
        + 0.08 * min(1.0, popularity * 100)
    )


def content_candidates(
    session: dict[str, Any],
    items: list[dict[str, Any]],
    impressions_by_session: dict[str, list[dict[str, Any]]],
    item_by_id: dict[str, dict[str, Any]],
    k: int,
) -> list[str]:
    profile = session_profile_from_history(session, impressions_by_session, item_by_id)
    scored = [
        (item["item_id"], content_similarity_score(profile, item))
        for item in items
        if str(item.get("available", "True")).lower() in {"true", "1"}
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scored[:k]]


def hybrid_candidates(
    session: dict[str, Any],
    items: list[dict[str, Any]],
    impressions_by_session: dict[str, list[dict[str, Any]]],
    item_by_id: dict[str, dict[str, Any]],
    k: int,
) -> list[str]:
    pop_k = max(10, k // 2)
    content_k = k
    pop = popularity_candidates(items, pop_k)
    content = content_candidates(session, items, impressions_by_session, item_by_id, content_k,)

    merged = []
    seen = set()
    for source in [content, pop]:
        for item_id in source:
            if item_id not in seen:
                merged.append(item_id)
                seen.add(item_id)
            if len(merged) >= k:
                break
        if len(merged) >= k:
            break

    if len(merged) < k:
        for item in sorted(items, key=lambda x: int(x["popularity_rank"])):
            if item["item_id"] not in seen:
                merged.append(item["item_id"])
                seen.add(item["item_id"])
            if len(merged) >= k:
                break

    return merged


def recall_at_k(candidate_sets: dict[str, list[str]], relevant_by_session: dict[str, set[str]], k: int) -> float:
    evaluated = 0
    hits = 0
    for session_id, relevant_items in relevant_by_session.items():
        if not relevant_items:
            continue
        evaluated += 1
        candidates = set(candidate_sets.get(session_id, [])[:k])
        if candidates & relevant_items:
            hits += 1
    if evaluated == 0:
        return 0.0
    return hits / evaluated


def coverage(candidate_sets: dict[str, list[str]], total_items: int) -> float:
    exposed = set()
    for candidates in candidate_sets.values():
        exposed.update(candidates)
    return len(exposed) / total_items if total_items else 0.0


def candidate_source_mix(candidate_sets: dict[str, list[str]], items: list[dict[str, Any]]) -> dict[str, int]:
    item_by_id = {x["item_id"]: x for x in items}
    categories = Counter()
    for candidates in candidate_sets.values():
        for item_id in candidates:
            item = item_by_id.get(item_id)
            if item:
                categories[item["category_l1"]] += 1
    return dict(categories.most_common())
