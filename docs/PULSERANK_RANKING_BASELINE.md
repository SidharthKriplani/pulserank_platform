# PulseRank Group 3 — Ranking Baseline

## Purpose

Group 3 adds a leakage-aware ranking baseline and offline evaluation harness.

The implementation uses a temporal split:

- train: Days 1–60
- holdout: Days 61–90

Random split is intentionally avoided because it can leak future popularity and behavioral signals into training.

## Generated Local Data

    data/processed/ranked_lists.csv

## Evidence Artifacts

    outputs/evidence/ranking_baseline_report.json
    outputs/evidence/offline_eval_log.json
    outputs/evidence/model_registry.json
    outputs/reports/group3_ranking_baseline_report_v1.json
    outputs/validation/ranking_baseline_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/run_ranking_baseline_v1.py
    PYTHONPATH=. python3 scripts/validate_ranking_baseline_v1.py

## Claim Boundary

This is an interpretable deterministic ranking baseline, not a production LightGBM deployment and not a neural ranker.
