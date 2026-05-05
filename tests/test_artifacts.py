"""
Smoke tests: verify all key evidence artifacts exist and have valid structure.
Run: pytest tests/test_artifacts.py -v
"""
import json
from pathlib import Path

ARTIFACTS = Path("artifacts")

def load(name):
    p = ARTIFACTS / name
    assert p.exists(), f"Missing artifact: {name}"
    with open(p) as f:
        return json.load(f)

def test_corpus_summary():
    d = load("corpus_summary.json")
    assert d["session_count"] >= 4000
    assert d["item_count"] >= 600
    assert d["impression_count"] >= 40000

def test_candidate_generation():
    d = load("candidate_generation_report.json")
    assert d["recall_at_100"] >= 0.60

def test_ranking_baseline():
    d = load("ranking_baseline_report.json")
    assert "ndcg_at_10" in d

def test_bias_correction():
    d = load("bias_correction_report.json")
    ips = d["ips_ndcg_at_10"]
    naive = d.get("naive_ndcg_at_10", d.get("biased_ndcg_at_10", 0))
    assert ips > naive, "IPS NDCG should exceed naive NDCG"

def test_diversity_report():
    d = load("diversity_report.json")
    assert "seller_gini" in d

def test_ab_simulation():
    d = load("ab_simulation_results.json")
    assert d["decision"] in ("HOLD_SIMULATED", "LAUNCH_SIMULATED")

def test_failure_scenarios():
    d = load("failure_recovery_report.json")
    scenarios = d.get("scenarios", d.get("failure_scenarios", []))
    assert len(scenarios) >= 10

def test_artifact_count():
    jsons = list(ARTIFACTS.glob("*.json"))
    assert len(jsons) >= 25, f"Expected ≥25 JSON artifacts, found {len(jsons)}"
