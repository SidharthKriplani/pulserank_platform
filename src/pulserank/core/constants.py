from __future__ import annotations

PROJECT_NAME = "PulseRank"
PROJECT_VERSION = "0.1.0"

REQUIRED_ARTIFACTS = [
    "outputs/evidence/corpus_summary.json",
    "outputs/evidence/candidate_generation_report.json",
    "outputs/evidence/ranking_baseline_report.json",
    "outputs/evidence/bias_correction_report.json",
    "outputs/evidence/diversity_guardrail_log.json",
    "outputs/evidence/conversion_attribution_report.json",
    "outputs/evidence/ab_simulation_results.json",
    "outputs/evidence/failure_recovery_report.json",
    "outputs/reports/pulserank_demo_report.json"
]

TRUTH_BOUNDARY_NOT_CLAIMED = [
    "real production deployment",
    "real users served",
    "real production traffic",
    "real online A/B test",
    "real revenue optimization",
    "real RL or contextual bandit optimization",
    "real ad auction system"
]
