from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


def parse_time(value: str) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def stable_variant(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return "treatment_governed_rerank" if bucket >= 5000 else "control_raw_ranker"


def build_purchase_index(purchases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index = defaultdict(list)
    for row in purchases:
        index[row["impression_id"]].append(row)
    return dict(index)


def attribute_impressions(
    impressions: list[dict[str, Any]],
    purchases: list[dict[str, Any]],
    window_days: int,
    return_window_days: int = 14,
) -> list[dict[str, Any]]:
    purchase_index = build_purchase_index(purchases)
    rows = []

    for imp in impressions:
        imp_ts = parse_time(imp["timestamp"])
        matched = None
        lag_hours = None

        for purchase in purchase_index.get(imp["impression_id"], []):
            purchase_ts = parse_time(purchase["purchase_timestamp"])
            if imp_ts is None or purchase_ts is None:
                continue

            lag = (purchase_ts - imp_ts).total_seconds() / 3600
            if 0 <= lag <= window_days * 24:
                matched = purchase
                lag_hours = lag
                break

        returned = 0
        return_observed_in_window = 0
        revenue = 0.0
        net_revenue = 0.0

        if matched is not None:
            returned = int(matched.get("returned", 0))
            revenue = float(matched.get("revenue", 0.0))
            net_revenue = revenue

            if returned:
                return_ts = parse_time(matched.get("return_timestamp", ""))
                purchase_ts = parse_time(matched.get("purchase_timestamp", ""))
                if return_ts and purchase_ts:
                    return_lag_days = (return_ts - purchase_ts).total_seconds() / 86400
                    if 0 <= return_lag_days <= return_window_days:
                        return_observed_in_window = 1
                        net_revenue = 0.0

        rows.append(
            {
                "impression_id": imp["impression_id"],
                "session_id": imp["session_id"],
                "user_id": imp["user_id"],
                "item_id": imp["item_id"],
                "display_rank": int(imp["display_rank"]),
                "window_days": window_days,
                "attributed_purchase": int(matched is not None),
                "attribution_lag_hours": round(lag_hours, 4) if lag_hours is not None else "",
                "attributed_revenue": round(revenue, 2),
                "returned": returned,
                "return_observed_in_window": return_observed_in_window,
                "net_revenue": round(net_revenue, 2),
                "data_gap_flag": False,
            }
        )

    return rows


def summarize_attribution(rows_by_window: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}

    for window, rows in rows_by_window.items():
        conversions = sum(int(r["attributed_purchase"]) for r in rows)
        revenue = sum(float(r["attributed_revenue"]) for r in rows)
        net_revenue = sum(float(r["net_revenue"]) for r in rows)
        returns = sum(int(r["return_observed_in_window"]) for r in rows)
        lags = [float(r["attribution_lag_hours"]) for r in rows if str(r["attribution_lag_hours"]).strip()]

        out[str(window)] = {
            "window_days": window,
            "rows": len(rows),
            "attributed_purchases": conversions,
            "attributed_cvr_per_impression": conversions / len(rows) if rows else 0.0,
            "attributed_revenue": revenue,
            "net_revenue_after_returns": net_revenue,
            "return_observed_count": returns,
            "return_rate_on_attributed_purchases": returns / conversions if conversions else 0.0,
            "mean_attribution_lag_hours": sum(lags) / len(lags) if lags else 0.0,
            "p90_attribution_lag_hours": percentile(lags, 0.90) if lags else 0.0,
        }

    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def seller_gini_from_lists(session_lists: dict[str, list[str]], item_by_id: dict[str, dict[str, Any]], k: int = 10) -> float:
    exposure = Counter()

    for items in session_lists.values():
        for item_id in items[:k]:
            if item_id in item_by_id:
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


def category_max_share_from_lists(session_lists: dict[str, list[str]], item_by_id: dict[str, dict[str, Any]], k: int = 10) -> float:
    shares = []

    for items in session_lists.values():
        top = items[:k]
        if not top:
            continue
        counts = Counter(item_by_id[item_id]["category_l1"] for item_id in top if item_id in item_by_id)
        if counts:
            shares.append(max(counts.values()) / len(top))

    return sum(shares) / len(shares) if shares else 0.0


def metric_lift(treatment: float, control: float) -> dict[str, float]:
    abs_lift = treatment - control
    rel_lift = abs_lift / control if control else 0.0
    return {
        "control": control,
        "treatment": treatment,
        "absolute_lift": abs_lift,
        "relative_lift": rel_lift,
    }
