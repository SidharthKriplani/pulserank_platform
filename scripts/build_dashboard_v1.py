from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(".")
EVIDENCE = ROOT / "outputs" / "evidence"
REPORTS = ROOT / "outputs" / "reports"
VALIDATION = ROOT / "outputs" / "validation"
DASHBOARD = ROOT / "outputs" / "dashboard"
DOCS = ROOT / "docs"


def load_json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def artifact_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.rglob("*.json")))


def status_badge(value: str) -> str:
    value = str(value)
    cls = "good" if value.lower() in {"pass", "ship_simulated"} else "warn" if "hold" in value.lower() or "review" in value.lower() else "info"
    return f'<span class="pill {cls}">{esc(value)}</span>'


def card(title: str, value, subtitle: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-title">{esc(title)}</div>
      <div class="metric-value">{esc(value)}</div>
      <div class="metric-subtitle">{esc(subtitle)}</div>
    </div>
    """


def table(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def main() -> None:
    corpus = load_json("outputs/evidence/corpus_summary.json")
    candidate = load_json("outputs/evidence/candidate_generation_report.json")
    ranking = load_json("outputs/evidence/ranking_baseline_report.json")
    bias = load_json("outputs/evidence/bias_correction_report.json")
    diversity = load_json("outputs/evidence/diversity_guardrail_log.json")
    attribution = load_json("outputs/evidence/conversion_attribution_report.json")
    ab = load_json("outputs/evidence/ab_simulation_results.json")
    failure = load_json("outputs/evidence/failure_recovery_report.json")
    demo = load_json("outputs/reports/pulserank_demo_report.json")

    evidence_count = artifact_count(EVIDENCE)
    validation_count = artifact_count(VALIDATION)
    report_count = artifact_count(REPORTS)
    total_artifacts = evidence_count + validation_count + report_count

    core = demo.get("core_numbers", {})
    recall = candidate.get("recall_at_100", {})
    before = diversity.get("before_rerank", {})
    after = diversity.get("after_rerank", {})
    ab_control = ab.get("control", {})
    ab_treatment = ab.get("treatment", {})
    primary_lift = ab.get("primary_metric_lift", {})

    failure_states = failure.get("failure_states", {})
    severity_mix = failure.get("severity_mix", {})

    metrics_html = "\n".join([
        card("Status", demo.get("status", "pass"), "final demo validation"),
        card("JSON artifacts", total_artifacts, f"{evidence_count} evidence / {validation_count} validation / {report_count} reports"),
        card("Sessions", corpus.get("sessions", "—"), "90-day synthetic marketplace corpus"),
        card("Impressions", corpus.get("impressions", "—"), "display-rank logged"),
        card("Display-rank coverage", corpus.get("display_rank_coverage", "—"), "required for IPS"),
        card("Hybrid Recall@100", recall.get("hybrid_popularity_content", "—"), "candidate generation"),
        card("IPS-weighted NDCG@10", bias.get("ips_weighted_ndcg_at_10", "—"), "position-bias corrected"),
        card("A/B decision", ab.get("decision", "—"), "offline simulation"),
        card("Failure scenarios", failure.get("scenario_count", "—"), "defense cases"),
        card("MetaSignal events", load_json("outputs/evidence/metasignal_integration_events.json").get("event_count", "—"), "schema-compatible export"),
        card("After seller Gini", after.get("seller_gini_at_10", "—"), "seller exposure governance"),
        card("After catalog coverage", after.get("catalog_coverage_at_10", "—"), "marketplace coverage"),
    ])

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PulseRank Evidence Dashboard</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: #0f1b2f;
      --panel2: #13233d;
      --text: #f8fafc;
      --muted: #a9b4c4;
      --line: rgba(255,255,255,.12);
      --blue: #3b82f6;
      --purple: #8b5cf6;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
      --cyan: #06b6d4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 15% 0%, rgba(59,130,246,.35), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(139,92,246,.32), transparent 32%),
        linear-gradient(135deg, #07111f 0%, #101c33 55%, #1e1b4b 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 48px 24px 72px;
    }}
    .hero {{
      border: 1px solid var(--line);
      background: rgba(15, 27, 47, .88);
      border-radius: 28px;
      padding: 36px;
      box-shadow: 0 24px 80px rgba(0,0,0,.32);
    }}
    .pills {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 22px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 8px 13px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .06em;
      text-transform: uppercase;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.06);
    }}
    .pill.good {{ color: #86efac; background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.35); }}
    .pill.warn {{ color: #fcd34d; background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.35); }}
    .pill.info {{ color: #bfdbfe; background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.35); }}
    .pill.purple {{ color: #ddd6fe; background: rgba(139,92,246,.16); border-color: rgba(139,92,246,.38); }}
    h1 {{
      margin: 0;
      font-size: clamp(42px, 7vw, 74px);
      line-height: .95;
      letter-spacing: -.06em;
    }}
    .lead {{
      max-width: 960px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.65;
      margin-top: 24px;
    }}
    .truth {{
      margin-top: 20px;
      color: #cbd5e1;
      font-size: 15px;
      line-height: 1.55;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-top: 24px;
    }}
    .metric-card, .section {{
      border: 1px solid var(--line);
      background: rgba(15, 27, 47, .86);
      border-radius: 20px;
      padding: 20px;
    }}
    .metric-title {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .12em;
      font-size: 12px;
      font-weight: 800;
      min-height: 30px;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 30px;
      font-weight: 900;
      letter-spacing: -.04em;
    }}
    .metric-subtitle {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .section {{
      margin-top: 24px;
      padding: 26px;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 26px;
      letter-spacing: -.03em;
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .flow-step {{
      background: rgba(255,255,255,.045);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 15px;
      color: #dbeafe;
      min-height: 112px;
    }}
    .flow-step b {{
      display: block;
      color: white;
      margin-bottom: 8px;
    }}
    .two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
    }}
    td {{
      padding: 13px 10px;
      border-bottom: 1px solid var(--line);
      color: #dbeafe;
      vertical-align: top;
    }}
    td:first-child {{
      color: var(--muted);
      font-weight: 800;
      width: 45%;
    }}
    .codebox {{
      background: #020617;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      color: #d1fae5;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      line-height: 1.55;
    }}
    .footer {{
      margin-top: 24px;
      color: var(--muted);
      text-align: center;
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .flow {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 620px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .flow {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="pills">
        {status_badge(demo.get("status", "pass"))}
        {status_badge(ab.get("decision", "HOLD_SIMULATED"))}
        <span class="pill info">IPS bias correction</span>
        <span class="pill purple">ranking governance</span>
        <span class="pill warn">offline A/B simulation</span>
      </div>
      <h1>PulseRank Evidence Dashboard</h1>
      <p class="lead">
        PulseRank is a production-simulated marketplace recommendation and ranking decision system.
        It demonstrates display-rank impression logging, hybrid candidate generation, temporal holdout ranking evaluation,
        IPS position-bias correction, delayed conversion attribution, seller/category exposure governance,
        offline A/B simulation, failure scenarios, and reproducible evidence artifacts.
      </p>
      <p class="truth">
        <b>Truth boundary:</b> solo-built, non-production, production-simulated. No real production deployment,
        no real users served, no real traffic, no real online A/B test, no real revenue optimization,
        and no real RL/contextual-bandit/ad-auction infrastructure.
      </p>
    </section>

    <section class="grid">
      {metrics_html}
    </section>

    <section class="section">
      <h2>System Flow</h2>
      <div class="flow">
        <div class="flow-step"><b>1. Corpus</b>90-day synthetic marketplace corpus with sessions, items, impressions, purchases.</div>
        <div class="flow-step"><b>2. Impression Logging</b>Every impression carries display_rank for IPS correction.</div>
        <div class="flow-step"><b>3. Candidate Generation</b>Popularity, content-similarity, and hybrid retrieval baseline.</div>
        <div class="flow-step"><b>4. Ranking</b>Temporal holdout baseline evaluation, not random split.</div>
        <div class="flow-step"><b>5. IPS Correction</b>Rank-level propensity estimation, clipping, and IPS-weighted evaluation.</div>
        <div class="flow-step"><b>6. Reranking</b>MMR-style diversity, seller caps, category exposure, novelty, coverage.</div>
        <div class="flow-step"><b>7. Attribution</b>Delayed conversion windows: 1/3/7 days with return-adjusted revenue.</div>
        <div class="flow-step"><b>8. Offline A/B</b>Control raw ranker vs treatment governed reranker.</div>
        <div class="flow-step"><b>9. Integration</b>MetaSignal-compatible metric event export.</div>
        <div class="flow-step"><b>10. Defense</b>Failure/recovery scenarios and final demo report.</div>
      </div>
    </section>

    <section class="two">
      <div class="section">
        <h2>Ranking + IPS Evidence</h2>
        {table([
          ("Candidate Recall@100", esc(recall.get("hybrid_popularity_content", "—"))),
          ("Holdout NDCG@10", esc(ranking.get("holdout_ndcg_at_10", "—"))),
          ("Raw NDCG@10", esc(bias.get("raw_ndcg_at_10", "—"))),
          ("IPS-weighted NDCG@10", esc(bias.get("ips_weighted_ndcg_at_10", "—"))),
          ("Bias delta", esc(bias.get("bias_delta_ips_minus_raw", "—"))),
          ("Propensity method", esc(bias.get("method", "—"))),
        ])}
      </div>
      <div class="section">
        <h2>Exposure Governance</h2>
        {table([
          ("Before seller Gini@10", esc(before.get("seller_gini_at_10", "—"))),
          ("After seller Gini@10", esc(after.get("seller_gini_at_10", "—"))),
          ("Before ILD@10", esc(before.get("mean_ild_at_10", "—"))),
          ("After ILD@10", esc(after.get("mean_ild_at_10", "—"))),
          ("Before catalog coverage@10", esc(before.get("catalog_coverage_at_10", "—"))),
          ("After catalog coverage@10", esc(after.get("catalog_coverage_at_10", "—"))),
        ])}
      </div>
    </section>

    <section class="two">
      <div class="section">
        <h2>Offline A/B Simulation</h2>
        {table([
          ("Decision", status_badge(ab.get("decision", "—"))),
          ("Control sessions", esc(ab_control.get("sessions", "—"))),
          ("Treatment sessions", esc(ab_treatment.get("sessions", "—"))),
          ("Control net RPS", esc(ab_control.get("net_revenue_per_session", "—"))),
          ("Treatment net RPS", esc(ab_treatment.get("net_revenue_per_session", "—"))),
          ("Primary absolute lift", esc(primary_lift.get("absolute_lift", "—"))),
          ("Guardrails pass", esc(ab.get("guardrails_pass", "—"))),
        ])}
      </div>
      <div class="section">
        <h2>Failure/Recovery Defense</h2>
        {table([
          ("Scenario count", esc(failure.get("scenario_count", "—"))),
          ("Failure states", esc(failure_states)),
          ("Severity mix", esc(severity_mix)),
          ("Critical boundary", "Offline simulation ≠ online A/B result"),
          ("Final report", "outputs/reports/pulserank_demo_report.json"),
        ])}
      </div>
    </section>

    <section class="section">
      <h2>Resume-Safe Claim</h2>
      <div class="codebox">{esc(demo.get("resume_claim", ""))}</div>
    </section>

    <section class="section">
      <h2>Run Locally</h2>
      <div class="codebox">PYTHONPATH=. python3 scripts/seed_demo.py
PYTHONPATH=. python3 scripts/run_candidate_generation_v1.py
PYTHONPATH=. python3 scripts/run_ranking_baseline_v1.py
PYTHONPATH=. python3 scripts/run_bias_correction_v1.py
PYTHONPATH=. python3 scripts/run_reranking_constraints_v1.py
PYTHONPATH=. python3 scripts/run_attribution_ab_v1.py
PYTHONPATH=. python3 scripts/show_demo_report.py</div>
    </section>

    <div class="footer">PulseRank Evidence Dashboard · static local/GitHub Pages artifact · generated from repo evidence</div>
  </main>
</body>
</html>
"""

    DASHBOARD.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    (DASHBOARD / "index.html").write_text(html_doc, encoding="utf-8")
    (DOCS / "index.html").write_text(html_doc, encoding="utf-8")

    manifest = {
        "artifact": "pulserank_dashboard_manifest_v1",
        "status": "pass",
        "dashboard_path": "outputs/dashboard/index.html",
        "github_pages_path": "docs/index.html",
        "evidence_json_count": evidence_count,
        "validation_json_count": validation_count,
        "report_json_count": report_count,
        "total_json_artifacts": total_artifacts,
        "source_reports": {
            "corpus": "outputs/evidence/corpus_summary.json",
            "candidate_generation": "outputs/evidence/candidate_generation_report.json",
            "ranking_baseline": "outputs/evidence/ranking_baseline_report.json",
            "bias_correction": "outputs/evidence/bias_correction_report.json",
            "diversity_guardrail": "outputs/evidence/diversity_guardrail_log.json",
            "attribution": "outputs/evidence/conversion_attribution_report.json",
            "ab_simulation": "outputs/evidence/ab_simulation_results.json",
            "failure_recovery": "outputs/evidence/failure_recovery_report.json",
            "demo": "outputs/reports/pulserank_demo_report.json",
        },
        "evidence_statement": "Static dashboard generated from committed PulseRank evidence artifacts for local review and GitHub Pages publishing."
    }

    (DASHBOARD / "dashboard_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("pulserank_dashboard_v1 complete")
    print("status: pass")
    print(f"total_json_artifacts: {total_artifacts}")
    print("wrote outputs/dashboard/index.html")
    print("wrote outputs/dashboard/dashboard_manifest.json")
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
