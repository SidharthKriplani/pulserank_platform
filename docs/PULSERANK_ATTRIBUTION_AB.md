# PulseRank Group 6 — Delayed Attribution and Offline A/B Simulation

## Purpose

Group 6 connects ranking outputs to business outcomes.

PulseRank now supports:

- delayed conversion attribution across 1/3/7-day windows
- return-adjusted net revenue
- deterministic offline A/B assignment
- control raw-ranker vs treatment governed-reranker comparison
- session-level CVR and revenue/session readout
- marketplace guardrails
- MetaSignal-compatible metric event export

## Generated Local Data

    data/processed/attributed_label_log_1d.csv
    data/processed/attributed_label_log_3d.csv
    data/processed/attributed_label_log_7d.csv
    data/processed/attributed_label_log_wide.csv
    data/processed/ab_assignment_log.csv
    data/processed/metasignal_integration_events.csv

## Evidence Artifacts

    outputs/evidence/conversion_attribution_report.json
    outputs/evidence/ab_simulation_results.json
    outputs/evidence/metasignal_integration_events.json
    outputs/reports/group6_attribution_ab_report_v1.json
    outputs/validation/attribution_ab_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/run_attribution_ab_v1.py
    PYTHONPATH=. python3 scripts/validate_attribution_ab_v1.py

## Claim Boundary

This is an offline simulation. It is not a real online A/B test and does not claim real revenue impact.
