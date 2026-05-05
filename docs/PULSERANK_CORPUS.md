# PulseRank Group 1 — Corpus and Schemas

## Purpose

Group 1 creates the deterministic marketplace corpus that every later PulseRank component depends on.

The most important schema decision is that every row in `impression_log` includes `display_rank`. Without display-rank logging, IPS position-bias correction cannot be implemented honestly.

## Generated Tables

Local reproducible data is written to:

    data/processed/user_sessions.csv
    data/processed/item_catalog.csv
    data/processed/impression_log.csv
    data/processed/purchase_log.csv
    data/processed/attributed_label_log.csv
    data/processed/exploration_log.csv
    data/processed/cold_start_events.csv

## Evidence Artifacts

Committed evidence summaries are written to:

    outputs/evidence/corpus_summary.json
    outputs/evidence/schema_manifest.json
    outputs/evidence/corpus_samples.json
    outputs/reports/group1_corpus_report_v1.json
    outputs/validation/corpus_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/seed_demo.py
    PYTHONPATH=. python3 scripts/validate_corpus_v1.py

## Claim Boundary

The corpus is synthetic and deterministic. It does not claim real marketplace traffic or real user behavior.
