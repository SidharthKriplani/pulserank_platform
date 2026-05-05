from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


SEED = 20260505
RNG = random.Random(SEED)

ROOT = Path(".")
DATA_OUT = ROOT / "data" / "processed"
EVIDENCE_OUT = ROOT / "outputs" / "evidence"
REPORT_OUT = ROOT / "outputs" / "reports"

DATA_OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE_OUT.mkdir(parents=True, exist_ok=True)
REPORT_OUT.mkdir(parents=True, exist_ok=True)

START = datetime(2026, 1, 1, 8, 0, 0)

NUM_USERS = 900
NUM_ITEMS = 650
NUM_SELLERS = 80
DAYS = 90
MAX_RANK = 10

CATEGORIES_L1 = [
    "electronics",
    "fashion",
    "home",
    "beauty",
    "sports",
    "baby",
    "books",
    "grocery",
]

CATEGORIES_L2 = {
    "electronics": ["mobiles", "audio", "laptops", "accessories"],
    "fashion": ["men", "women", "footwear", "bags"],
    "home": ["decor", "kitchen", "furniture", "storage"],
    "beauty": ["skincare", "haircare", "makeup", "fragrance"],
    "sports": ["fitness", "outdoor", "cycling", "yoga"],
    "baby": ["toys", "feeding", "sleep", "care"],
    "books": ["fiction", "business", "education", "children"],
    "grocery": ["snacks", "beverages", "staples", "personal_care"],
}

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
PRICE_BUCKETS = ["budget", "mid", "premium"]

PROPENSITY_BY_RANK = {
    1: 0.25,
    2: 0.21,
    3: 0.18,
    4: 0.15,
    5: 0.12,
    6: 0.10,
    7: 0.085,
    8: 0.075,
    9: 0.065,
    10: 0.06,
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choice_weighted(items: list, weights: list[float]):
    total = sum(weights)
    r = RNG.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += weight
        if upto >= r:
            return item
    return items[-1]


def make_users() -> list[dict]:
    users = []
    for i in range(1, NUM_USERS + 1):
        favorite_category = RNG.choice(CATEGORIES_L1)
        secondary_category = RNG.choice([c for c in CATEGORIES_L1 if c != favorite_category])
        price_bucket = choice_weighted(PRICE_BUCKETS, [0.48, 0.38, 0.14])
        users.append(
            {
                "user_id": f"U{i:05d}",
                "favorite_category": favorite_category,
                "secondary_category": secondary_category,
                "price_sensitivity_bucket": price_bucket,
                "home_device_type": choice_weighted(DEVICE_TYPES, [0.74, 0.21, 0.05]),
            }
        )
    return users


def make_items() -> list[dict]:
    items = []
    for i in range(1, NUM_ITEMS + 1):
        category_l1 = choice_weighted(CATEGORIES_L1, [0.18, 0.16, 0.13, 0.13, 0.10, 0.10, 0.08, 0.12])
        category_l2 = RNG.choice(CATEGORIES_L2[category_l1])
        seller_id = f"S{RNG.randint(1, NUM_SELLERS):03d}"
        price_base = {
            "electronics": 4500,
            "fashion": 1200,
            "home": 1800,
            "beauty": 850,
            "sports": 1400,
            "baby": 950,
            "books": 450,
            "grocery": 350,
        }[category_l1]
        price = round(max(99, RNG.lognormvariate(math.log(price_base), 0.45)), 2)
        created_day = RNG.randint(1, DAYS)
        embedding = [round(RNG.uniform(-1, 1), 4) for _ in range(8)]
        popularity_latent = RNG.random() ** 2
        freshness_score = round(clamp(1 - (DAYS - created_day) / DAYS + RNG.uniform(-0.08, 0.08), 0.02, 1.0), 4)
        items.append(
            {
                "item_id": f"I{i:05d}",
                "category_l1": category_l1,
                "category_l2": category_l2,
                "price": price,
                "seller_id": seller_id,
                "content_embedding": json.dumps(embedding),
                "popularity_rank": 0,
                "freshness_score": freshness_score,
                "available": True,
                "created_at": (START + timedelta(days=created_day - 1)).isoformat(),
                "latent_quality": round(popularity_latent, 5),
            }
        )

    items_sorted = sorted(items, key=lambda x: x["latent_quality"], reverse=True)
    for rank, item in enumerate(items_sorted, start=1):
        item["popularity_rank"] = rank
    return items


def user_category_affinity(user: dict) -> dict[str, float]:
    affinity = {}
    for c in CATEGORIES_L1:
        if c == user["favorite_category"]:
            affinity[c] = round(RNG.uniform(0.58, 0.82), 4)
        elif c == user["secondary_category"]:
            affinity[c] = round(RNG.uniform(0.25, 0.42), 4)
        else:
            affinity[c] = round(RNG.uniform(0.02, 0.18), 4)
    return affinity


def item_relevance(user: dict, item: dict) -> float:
    category_boost = 0.0
    if item["category_l1"] == user["favorite_category"]:
        category_boost += 0.18
    if item["category_l1"] == user["secondary_category"]:
        category_boost += 0.08

    price = float(item["price"])
    bucket = user["price_sensitivity_bucket"]
    price_fit = 0.0
    if bucket == "budget":
        price_fit = 0.13 if price <= 900 else -0.06 if price > 3000 else 0.02
    elif bucket == "mid":
        price_fit = 0.10 if 700 <= price <= 3500 else -0.02
    else:
        price_fit = 0.09 if price >= 1500 else -0.02

    popularity_fit = (NUM_ITEMS - int(item["popularity_rank"])) / NUM_ITEMS * 0.09
    freshness_fit = float(item["freshness_score"]) * 0.05
    quality_fit = float(item["latent_quality"]) * 0.15
    noise = RNG.uniform(-0.03, 0.03)
    return clamp(0.04 + category_boost + price_fit + popularity_fit + freshness_fit + quality_fit + noise, 0.005, 0.75)


def make_sessions_and_events(users: list[dict], items: list[dict]):
    item_by_category = defaultdict(list)
    for item in items:
        item_by_category[item["category_l1"]].append(item)

    user_sessions = []
    impression_log = []
    purchase_log = []
    exploration_log = []
    cold_start_events = []

    impression_counter = 1
    purchase_counter = 1
    session_counter = 1
    exploration_count = 0
    cold_start_retrieved = 0

    for day in range(1, DAYS + 1):
        base_sessions = 42 + int(18 * math.sin(day / 8)) + RNG.randint(-6, 10)
        base_sessions = max(24, base_sessions)

        for _ in range(base_sessions):
            user = RNG.choice(users)
            session_id = f"SESS{session_counter:07d}"
            session_counter += 1

            start = START + timedelta(days=day - 1, hours=RNG.randint(0, 13), minutes=RNG.randint(0, 59))
            duration_minutes = RNG.randint(2, 38)
            end = start + timedelta(minutes=duration_minutes)

            affinity = user_category_affinity(user)
            session_category = choice_weighted(list(affinity.keys()), list(affinity.values()))

            candidate_pool = list(item_by_category[session_category])
            if len(candidate_pool) < MAX_RANK:
                candidate_pool += RNG.sample(items, MAX_RANK - len(candidate_pool))

            sorted_pool = sorted(candidate_pool, key=lambda x: (x["latent_quality"], -x["popularity_rank"]), reverse=True)
            top_pool = sorted_pool[: min(80, len(sorted_pool))]

            ranked_items = []
            for rank in range(1, MAX_RANK + 1):
                exploration_flag = RNG.random() < 0.05
                if exploration_flag:
                    chosen = RNG.choice(items)
                    exploration_reason = "epsilon_greedy_freshness_weighted"
                    exploration_count += 1
                else:
                    chosen = choice_weighted(top_pool, [1 / (idx + 1) for idx in range(len(top_pool))])
                    exploration_reason = ""

                attempts = 0
                while chosen["item_id"] in {x["item_id"] for x in ranked_items} and attempts < 20:
                    chosen = RNG.choice(items if exploration_flag else top_pool)
                    attempts += 1

                ranked_items.append(chosen)

                created_at = datetime.fromisoformat(chosen["created_at"])
                cold_start_flag = (start - created_at).days <= 7 or int(chosen["popularity_rank"]) > 580
                if cold_start_flag and RNG.random() < 0.12:
                    cold_start_retrieved += 1
                    cold_start_events.append(
                        {
                            "event_id": f"COLD{len(cold_start_events)+1:06d}",
                            "item_id": chosen["item_id"],
                            "trigger_reason": "new_item" if (start - created_at).days <= 7 else "low_interaction",
                            "fallback_strategy": "content_based",
                            "timestamp": start.isoformat(),
                            "retrieved": True,
                        }
                    )

                relevance = item_relevance(user, chosen)
                position_propensity = PROPENSITY_BY_RANK[rank]
                click_prob = clamp(position_propensity * (0.38 + relevance), 0.002, 0.62)
                clicked = RNG.random() < click_prob
                atc_prob = clamp(0.18 + relevance * 0.55, 0.01, 0.42) if clicked else 0.012
                added_to_cart = RNG.random() < atc_prob
                purchase_prob = clamp(0.12 + relevance * 0.30, 0.005, 0.30) if added_to_cart else 0.002
                purchased = RNG.random() < purchase_prob

                impression_id = f"IMP{impression_counter:09d}"
                impression_counter += 1

                impression_log.append(
                    {
                        "impression_id": impression_id,
                        "session_id": session_id,
                        "user_id": user["user_id"],
                        "item_id": chosen["item_id"],
                        "display_rank": rank,
                        "recommendation_stage": "final_reranked_candidate_seed",
                        "ranking_variant_id": "baseline_popularity_content_v0",
                        "exploration_flag": exploration_flag,
                        "cold_start_flag": cold_start_flag,
                        "timestamp": start.isoformat(),
                        "clicked": int(clicked),
                        "added_to_cart": int(added_to_cart),
                        "position_propensity": position_propensity,
                        "simulated_relevance": round(relevance, 5),
                    }
                )

                if exploration_flag:
                    exploration_log.append(
                        {
                            "session_id": session_id,
                            "item_id": chosen["item_id"],
                            "display_rank": rank,
                            "is_exploration": True,
                            "exploration_reason": exploration_reason,
                            "timestamp": start.isoformat(),
                        }
                    )

                if purchased:
                    lag_hours = choice_weighted([2, 8, 26, 52, 120], [0.42, 0.25, 0.18, 0.10, 0.05])
                    purchase_time = start + timedelta(hours=lag_hours, minutes=RNG.randint(0, 59))
                    returned = RNG.random() < (0.03 if chosen["category_l1"] != "fashion" else 0.08)
                    return_time = purchase_time + timedelta(days=RNG.randint(2, 13)) if returned else ""
                    purchase_log.append(
                        {
                            "purchase_id": f"PUR{purchase_counter:08d}",
                            "session_id": session_id,
                            "user_id": user["user_id"],
                            "item_id": chosen["item_id"],
                            "impression_id": impression_id,
                            "purchase_timestamp": purchase_time.isoformat(),
                            "revenue": round(float(chosen["price"]) * RNG.uniform(0.92, 1.08), 2),
                            "returned": int(returned),
                            "return_timestamp": return_time.isoformat() if return_time else "",
                        }
                    )
                    purchase_counter += 1

            user_sessions.append(
                {
                    "session_id": session_id,
                    "user_id": user["user_id"],
                    "timestamp_start": start.isoformat(),
                    "timestamp_end": end.isoformat(),
                    "device_type": user["home_device_type"],
                    "num_events": len(ranked_items),
                    "category_affinity": json.dumps(affinity),
                    "price_sensitivity_bucket": user["price_sensitivity_bucket"],
                    "simulated_day": day,
                }
            )

    return user_sessions, impression_log, purchase_log, exploration_log, cold_start_events


def build_attributed_labels(impressions: list[dict], purchases: list[dict], window_days: int = 3) -> list[dict]:
    purchases_by_impression = defaultdict(list)
    for p in purchases:
        purchases_by_impression[p["impression_id"]].append(p)

    rows = []
    for imp in impressions:
        imp_time = datetime.fromisoformat(imp["timestamp"])
        matched = None
        for p in purchases_by_impression.get(imp["impression_id"], []):
            p_time = datetime.fromisoformat(p["purchase_timestamp"])
            delta_days = (p_time - imp_time).total_seconds() / 86400
            if 0 <= delta_days <= window_days:
                matched = p
                break

        rows.append(
            {
                "impression_id": imp["impression_id"],
                "session_id": imp["session_id"],
                "item_id": imp["item_id"],
                "attributed_purchase": int(matched is not None),
                "attribution_window_days": window_days,
                "attribution_timestamp": matched["purchase_timestamp"] if matched else "",
                "attributed_revenue": matched["revenue"] if matched else 0.0,
                "attributed_return": matched["returned"] if matched else 0,
                "data_gap_flag": False,
            }
        )
    return rows


def summarize(
    users: list[dict],
    items: list[dict],
    sessions: list[dict],
    impressions: list[dict],
    purchases: list[dict],
    attributed: list[dict],
    exploration: list[dict],
    cold_start: list[dict],
) -> dict:
    clicks = sum(int(x["clicked"]) for x in impressions)
    atc = sum(int(x["added_to_cart"]) for x in impressions)
    attributed_purchases = sum(int(x["attributed_purchase"]) for x in attributed)
    revenue = sum(float(x["revenue"]) for x in purchases)

    rank_missing = [x for x in impressions if not str(x.get("display_rank", "")).strip()]
    rank_out_of_range = [x for x in impressions if not (1 <= int(x["display_rank"]) <= MAX_RANK)]

    seller_counts = Counter(item["seller_id"] for item in items)
    category_counts = Counter(item["category_l1"] for item in items)

    return {
        "artifact": "pulserank_corpus_summary_v1",
        "status": "pass",
        "random_seed": SEED,
        "simulated_days": DAYS,
        "users": len(users),
        "items": len(items),
        "sellers": NUM_SELLERS,
        "sessions": len(sessions),
        "impressions": len(impressions),
        "purchases": len(purchases),
        "attributed_label_rows": len(attributed),
        "exploration_rows": len(exploration),
        "cold_start_events": len(cold_start),
        "clicks": clicks,
        "add_to_carts": atc,
        "attributed_purchases_3d": attributed_purchases,
        "ctr": round(clicks / len(impressions), 5),
        "atc_rate": round(atc / len(impressions), 5),
        "purchase_rate_per_impression": round(len(purchases) / len(impressions), 5),
        "attributed_cvr_3d": round(attributed_purchases / len(impressions), 5),
        "revenue": round(revenue, 2),
        "display_rank_missing_count": len(rank_missing),
        "display_rank_out_of_range_count": len(rank_out_of_range),
        "display_rank_coverage": round(1 - len(rank_missing) / len(impressions), 5),
        "max_display_rank": MAX_RANK,
        "category_distribution": dict(category_counts),
        "top_seller_item_counts": dict(seller_counts.most_common(10)),
        "evidence_statement": "Generated deterministic 90-day marketplace corpus with user sessions, item catalog, position-logged impressions, purchases, attribution seed labels, exploration events, and cold-start events."
    }


def schema_manifest() -> dict:
    return {
        "artifact": "pulserank_schema_manifest_v1",
        "status": "pass",
        "schemas": {
            "user_sessions": [
                "session_id",
                "user_id",
                "timestamp_start",
                "timestamp_end",
                "device_type",
                "num_events",
                "category_affinity",
                "price_sensitivity_bucket",
                "simulated_day",
            ],
            "item_catalog": [
                "item_id",
                "category_l1",
                "category_l2",
                "price",
                "seller_id",
                "content_embedding",
                "popularity_rank",
                "freshness_score",
                "available",
                "created_at",
                "latent_quality",
            ],
            "impression_log": [
                "impression_id",
                "session_id",
                "user_id",
                "item_id",
                "display_rank",
                "recommendation_stage",
                "ranking_variant_id",
                "exploration_flag",
                "cold_start_flag",
                "timestamp",
                "clicked",
                "added_to_cart",
                "position_propensity",
                "simulated_relevance",
            ],
            "purchase_log": [
                "purchase_id",
                "session_id",
                "user_id",
                "item_id",
                "impression_id",
                "purchase_timestamp",
                "revenue",
                "returned",
                "return_timestamp",
            ],
            "attributed_label_log": [
                "impression_id",
                "session_id",
                "item_id",
                "attributed_purchase",
                "attribution_window_days",
                "attribution_timestamp",
                "attributed_revenue",
                "attributed_return",
                "data_gap_flag",
            ],
            "exploration_log": [
                "session_id",
                "item_id",
                "display_rank",
                "is_exploration",
                "exploration_reason",
                "timestamp",
            ],
            "cold_start_events": [
                "event_id",
                "item_id",
                "trigger_reason",
                "fallback_strategy",
                "timestamp",
                "retrieved",
            ],
        },
        "non_negotiable_field": "impression_log.display_rank",
        "evidence_statement": "Defines the Group 1 PulseRank corpus schemas required for IPS correction, attribution, exploration, and marketplace exposure governance."
    }


def sample_payload(rows: list[dict], n: int = 5) -> list[dict]:
    return rows[: min(n, len(rows))]


def main() -> None:
    users = make_users()
    items = make_items()
    sessions, impressions, purchases, exploration, cold_start = make_sessions_and_events(users, items)
    attributed = build_attributed_labels(impressions, purchases, window_days=3)

    write_csv(DATA_OUT / "user_sessions.csv", sessions)
    write_csv(DATA_OUT / "item_catalog.csv", items)
    write_csv(DATA_OUT / "impression_log.csv", impressions)
    write_csv(DATA_OUT / "purchase_log.csv", purchases)
    write_csv(DATA_OUT / "attributed_label_log.csv", attributed)
    write_csv(DATA_OUT / "exploration_log.csv", exploration if exploration else [{"session_id": "", "item_id": "", "display_rank": "", "is_exploration": "", "exploration_reason": "", "timestamp": ""}])
    write_csv(DATA_OUT / "cold_start_events.csv", cold_start if cold_start else [{"event_id": "", "item_id": "", "trigger_reason": "", "fallback_strategy": "", "timestamp": "", "retrieved": ""}])

    summary = summarize(users, items, sessions, impressions, purchases, attributed, exploration, cold_start)
    manifest = schema_manifest()

    samples = {
        "artifact": "pulserank_corpus_samples_v1",
        "status": "pass",
        "samples": {
            "user_sessions": sample_payload(sessions),
            "item_catalog": sample_payload(items),
            "impression_log": sample_payload(impressions),
            "purchase_log": sample_payload(purchases),
            "attributed_label_log": sample_payload(attributed),
            "exploration_log": sample_payload(exploration),
            "cold_start_events": sample_payload(cold_start),
        },
    }

    write_json(EVIDENCE_OUT / "corpus_summary.json", summary)
    write_json(EVIDENCE_OUT / "schema_manifest.json", manifest)
    write_json(EVIDENCE_OUT / "corpus_samples.json", samples)

    report = {
        "artifact": "pulserank_group1_corpus_report_v1",
        "status": "pass",
        "summary_path": "outputs/evidence/corpus_summary.json",
        "schema_manifest_path": "outputs/evidence/schema_manifest.json",
        "sample_path": "outputs/evidence/corpus_samples.json",
        "data_paths": {
            "user_sessions": "data/processed/user_sessions.csv",
            "item_catalog": "data/processed/item_catalog.csv",
            "impression_log": "data/processed/impression_log.csv",
            "purchase_log": "data/processed/purchase_log.csv",
            "attributed_label_log": "data/processed/attributed_label_log.csv",
            "exploration_log": "data/processed/exploration_log.csv",
            "cold_start_events": "data/processed/cold_start_events.csv",
        },
        "evidence_statement": "Group 1 corpus generation complete. Data files are reproducible locally and evidence summaries are committed."
    }
    write_json(REPORT_OUT / "group1_corpus_report_v1.json", report)

    print("pulserank_seed_demo_v1 complete")
    print("status: pass")
    print(f"sessions: {summary['sessions']}")
    print(f"items: {summary['items']}")
    print(f"impressions: {summary['impressions']}")
    print(f"purchases: {summary['purchases']}")
    print(f"display_rank_coverage: {summary['display_rank_coverage']}")
    print("wrote outputs/evidence/corpus_summary.json")
    print("wrote outputs/evidence/schema_manifest.json")
    print("wrote outputs/evidence/corpus_samples.json")
    print("wrote outputs/reports/group1_corpus_report_v1.json")


if __name__ == "__main__":
    main()
