from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.pulserank.evaluation.attribution import (
    attribute_impressions,
    category_max_share_from_lists,
    metric_lift,
    seller_gini_from_lists,
    stable_variant,
    summarize_attribution,
)


ROOT = Path(".")
DATA = ROOT / "data" / "processed"
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"

WINDOWS = [1, 3, 7]
DEFAULT_WINDOW = 3
TOP_K = 10


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


def ranked_lists_from_rows(rows: list[dict], position_col: str) -> dict[str, list[str]]:
    out = defaultdict(list)
    sorted_rows = sorted(rows, key=lambda r: (r["session_id"], int(r[position_col])))
    for row in sorted_rows:
        out[row["session_id"]].append(row["item_id"])
    return dict(out)


def session_purchase_map(attribution_rows: list[dict]) -> dict[str, dict[str, dict]]:
    out = defaultdict(dict)
    for row in attribution_rows:
        if int(row["attributed_purchase"]) == 1:
            out[row["session_id"]][row["item_id"]] = row
    return dict(out)


def summarize_variant(
    variant_name: str,
    assigned_sessions: list[str],
    session_lists: dict[str, list[str]],
    purchase_map: dict[str, dict[str, dict]],
    item_by_id: dict[str, dict],
) -> dict:
    sessions = 0
    converted_sessions = 0
    revenue = 0.0
    net_revenue = 0.0
    returns = 0
    top10_impressions = 0

    used_lists = {}

    for session_id in assigned_sessions:
        items = session_lists.get(session_id, [])[:TOP_K]
        if not items:
            continue

        sessions += 1
        top10_impressions += len(items)
        used_lists[session_id] = items

        matched_items = []
        for item_id in items:
            attr = purchase_map.get(session_id, {}).get(item_id)
            if attr:
                matched_items.append(attr)

        if matched_items:
            converted_sessions += 1
            revenue += sum(float(x["attributed_revenue"]) for x in matched_items)
            net_revenue += sum(float(x["net_revenue"]) for x in matched_items)
            returns += sum(int(x["return_observed_in_window"]) for x in matched_items)

    cvr = converted_sessions / sessions if sessions else 0.0
    rps = revenue / sessions if sessions else 0.0
    net_rps = net_revenue / sessions if sessions else 0.0
    return_rate = returns / converted_sessions if converted_sessions else 0.0

    return {
        "variant": variant_name,
        "sessions": sessions,
        "top10_impressions": top10_impressions,
        "converted_sessions": converted_sessions,
        "session_cvr": cvr,
        "revenue": revenue,
        "revenue_per_session": rps,
        "net_revenue_after_returns": net_revenue,
        "net_revenue_per_session": net_rps,
        "return_rate_on_converted_sessions": return_rate,
        "seller_gini_at_10": seller_gini_from_lists(used_lists, item_by_id, TOP_K),
        "mean_category_max_share_at_10": category_max_share_from_lists(used_lists, item_by_id, TOP_K),
    }


def make_metasignal_events(ab_results: dict, attribution_report: dict, generated_at: str = "2026-05-05T00:00:00") -> list[dict]:
    rows = []
    event_counter = 1
    experiment_id = "pulserank_offline_ab_v1"

    for variant_key in ["control", "treatment"]:
        variant = ab_results[variant_key]
        for metric in [
            "session_cvr",
            "revenue_per_session",
            "net_revenue_per_session",
            "return_rate_on_converted_sessions",
            "seller_gini_at_10",
            "mean_category_max_share_at_10",
        ]:
            rows.append(
                {
                    "event_id": f"MSIG{event_counter:06d}",
                    "source_project": "PulseRank",
                    "experiment_id": experiment_id,
                    "variant": variant["variant"],
                    "metric_name": metric,
                    "metric_value": round(float(variant[metric]), 8),
                    "unit_of_analysis": "session",
                    "timestamp": generated_at,
                    "claim_boundary": "offline_simulation_not_online_experiment",
                }
            )
            event_counter += 1

    for window, summary in attribution_report["window_summaries"].items():
        rows.append(
            {
                "event_id": f"MSIG{event_counter:06d}",
                "source_project": "PulseRank",
                "experiment_id": experiment_id,
                "variant": "attribution_monitor",
                "metric_name": f"attributed_cvr_{window}d",
                "metric_value": round(float(summary["attributed_cvr_per_impression"]), 8),
                "unit_of_analysis": "impression",
                "timestamp": generated_at,
                "claim_boundary": "offline_simulation_not_online_experiment",
            }
        )
        event_counter += 1

    return rows


def main() -> None:
    sessions = read_csv(DATA / "user_sessions.csv")
    items = read_csv(DATA / "item_catalog.csv")
    impressions = read_csv(DATA / "impression_log.csv")
    purchases = read_csv(DATA / "purchase_log.csv")
    ranked = read_csv(DATA / "ranked_lists.csv")
    reranked = read_csv(DATA / "reranked_lists.csv")

    item_by_id = {row["item_id"]: row for row in items}
    session_day = {row["session_id"]: int(row["simulated_day"]) for row in sessions}
    holdout_sessions = [sid for sid, day in session_day.items() if day > 60]

    rows_by_window = {}
    wide_rows = []

    for window in WINDOWS:
        attributed = attribute_impressions(impressions, purchases, window_days=window, return_window_days=14)
        rows_by_window[window] = attributed

        write_csv(DATA / f"attributed_label_log_{window}d.csv", attributed)

    for base in rows_by_window[DEFAULT_WINDOW]:
        merged = dict(base)
        for window in WINDOWS:
            match = rows_by_window[window][len(wide_rows)]
            merged[f"attributed_purchase_{window}d"] = match["attributed_purchase"]
            merged[f"attributed_revenue_{window}d"] = match["attributed_revenue"]
            merged[f"net_revenue_{window}d"] = match["net_revenue"]
        wide_rows.append(merged)

    write_csv(DATA / "attributed_label_log_wide.csv", wide_rows)

    window_summaries = summarize_attribution(rows_by_window)

    attribution_report = {
        "artifact": "pulserank_conversion_attribution_report_v1",
        "status": "pass",
        "attribution_windows_days": WINDOWS,
        "default_window_days": DEFAULT_WINDOW,
        "return_window_days": 14,
        "impressions_evaluated": len(impressions),
        "purchases_evaluated": len(purchases),
        "window_summaries": {
            str(k): {
                kk: round(vv, 5) if isinstance(vv, float) else vv
                for kk, vv in summary.items()
            }
            for k, summary in window_summaries.items()
        },
        "data_paths": {
            "attributed_label_log_1d": "data/processed/attributed_label_log_1d.csv",
            "attributed_label_log_3d": "data/processed/attributed_label_log_3d.csv",
            "attributed_label_log_7d": "data/processed/attributed_label_log_7d.csv",
            "attributed_label_log_wide": "data/processed/attributed_label_log_wide.csv",
        },
        "evidence_statement": "Implements delayed conversion attribution with 1/3/7-day sensitivity windows and return-adjusted net revenue."
    }

    default_attribution = rows_by_window[DEFAULT_WINDOW]
    purchase_map = session_purchase_map(default_attribution)

    control_lists = ranked_lists_from_rows(ranked, "model_rank")
    treatment_lists = ranked_lists_from_rows(reranked, "reranked_position")

    control_sessions = []
    treatment_sessions = []
    assignment_rows = []

    for sid in holdout_sessions:
        variant = stable_variant(sid)
        if variant == "control_raw_ranker":
            control_sessions.append(sid)
        else:
            treatment_sessions.append(sid)

        assignment_rows.append(
            {
                "experiment_id": "pulserank_offline_ab_v1",
                "session_id": sid,
                "variant": variant,
                "assignment_unit": "session",
                "assignment_method": "deterministic_sha256_half_split",
            }
        )

    write_csv(DATA / "ab_assignment_log.csv", assignment_rows)

    control = summarize_variant("control_raw_ranker", control_sessions, control_lists, purchase_map, item_by_id)
    treatment = summarize_variant("treatment_governed_rerank", treatment_sessions, treatment_lists, purchase_map, item_by_id)

    cvr_lift = metric_lift(treatment["session_cvr"], control["session_cvr"])
    rps_lift = metric_lift(treatment["revenue_per_session"], control["revenue_per_session"])
    net_rps_lift = metric_lift(treatment["net_revenue_per_session"], control["net_revenue_per_session"])

    guardrails = {
        "seller_gini_treatment_lte_0_65": treatment["seller_gini_at_10"] <= 0.65,
        "category_max_share_treatment_lte_0_55": treatment["mean_category_max_share_at_10"] <= 0.55,
        "return_rate_not_more_than_control_plus_3pp": treatment["return_rate_on_converted_sessions"] <= control["return_rate_on_converted_sessions"] + 0.03,
        "minimum_treatment_sessions": treatment["sessions"] >= 250,
    }

    primary_pass = net_rps_lift["absolute_lift"] > 0
    guardrails_pass = all(guardrails.values())
    decision = "SHIP_SIMULATED" if primary_pass and guardrails_pass else "HOLD_SIMULATED"

    ab_results = {
        "artifact": "pulserank_ab_simulation_results_v1",
        "status": "pass",
        "experiment_id": "pulserank_offline_ab_v1",
        "simulation_type": "offline_temporal_holdout_session_split",
        "assignment_method": "deterministic_sha256_half_split",
        "control": {k: round(v, 5) if isinstance(v, float) else v for k, v in control.items()},
        "treatment": {k: round(v, 5) if isinstance(v, float) else v for k, v in treatment.items()},
        "primary_metric": "net_revenue_per_session",
        "primary_metric_lift": {
            k: round(v, 5) if isinstance(v, float) else v
            for k, v in net_rps_lift.items()
        },
        "supporting_metric_lifts": {
            "session_cvr": {k: round(v, 5) if isinstance(v, float) else v for k, v in cvr_lift.items()},
            "revenue_per_session": {k: round(v, 5) if isinstance(v, float) else v for k, v in rps_lift.items()},
        },
        "guardrails": guardrails,
        "guardrails_pass": guardrails_pass,
        "decision": decision,
        "interpretation": "Offline A/B simulation is evidence for portfolio demonstration only; it is not an online experiment result.",
        "evidence_statement": "Compares raw ranking vs governed reranking on temporal holdout sessions using delayed-attribution labels and marketplace guardrails."
    }

    metasignal_events = make_metasignal_events(ab_results, attribution_report)
    write_csv(DATA / "metasignal_integration_events.csv", metasignal_events)

    metasignal_report = {
        "artifact": "pulserank_metasignal_integration_events_v1",
        "status": "pass",
        "event_count": len(metasignal_events),
        "schema": list(metasignal_events[0].keys()),
        "data_path": "data/processed/metasignal_integration_events.csv",
        "compatible_target": "MetaSignal-style experiment metric event ingestion",
        "claim_boundary": "schema-compatible export only; no live MetaSignal ingestion performed",
        "events_sample": metasignal_events[:8],
    }

    write_json(EVIDENCE / "conversion_attribution_report.json", attribution_report)
    write_json(EVIDENCE / "ab_simulation_results.json", ab_results)
    write_json(EVIDENCE / "metasignal_integration_events.json", metasignal_report)

    group_report = {
        "artifact": "pulserank_group6_attribution_ab_report_v1",
        "status": "pass",
        "conversion_attribution_report_path": "outputs/evidence/conversion_attribution_report.json",
        "ab_simulation_results_path": "outputs/evidence/ab_simulation_results.json",
        "metasignal_integration_events_path": "outputs/evidence/metasignal_integration_events.json",
        "evidence_statement": "Group 6 attribution and A/B simulation complete with delayed conversion labels, return-adjusted revenue, offline experiment readout, and MetaSignal-compatible event export."
    }
    write_json(REPORTS / "group6_attribution_ab_report_v1.json", group_report)

    print("pulserank_attribution_ab_v1 complete")
    print("status: pass")
    print(f"impressions_evaluated: {len(impressions)}")
    print(f"purchases_evaluated: {len(purchases)}")
    print(f"control_sessions: {control['sessions']}")
    print(f"treatment_sessions: {treatment['sessions']}")
    print(f"control_net_rps: {ab_results['control']['net_revenue_per_session']}")
    print(f"treatment_net_rps: {ab_results['treatment']['net_revenue_per_session']}")
    print(f"primary_metric_absolute_lift: {ab_results['primary_metric_lift']['absolute_lift']}")
    print(f"decision: {decision}")
    print(f"metasignal_event_count: {len(metasignal_events)}")
    print("wrote outputs/evidence/conversion_attribution_report.json")
    print("wrote outputs/evidence/ab_simulation_results.json")
    print("wrote outputs/evidence/metasignal_integration_events.json")
    print("wrote outputs/reports/group6_attribution_ab_report_v1.json")


if __name__ == "__main__":
    main()
