# PulseRank Group 2 — Candidate Generation

## Purpose

Candidate generation reduces the full catalog into a smaller candidate set before ranking.

The must-have implementation includes:

- popularity baseline
- content-similarity baseline
- hybrid popularity + content candidate generation

The staff-level point is retrieval recall discipline: items not retrieved cannot be ranked.

## Generated Local Data

    data/processed/candidate_sets.csv

## Evidence Artifacts

    outputs/evidence/candidate_generation_report.json
    outputs/evidence/candidate_recall_report.json
    outputs/evidence/candidate_samples.json
    outputs/reports/group2_candidate_generation_report_v1.json
    outputs/validation/candidate_generation_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/run_candidate_generation_v1.py
    PYTHONPATH=. python3 scripts/validate_candidate_generation_v1.py

## Claim Boundary

This is baseline candidate generation, not a two-tower neural retrieval model and not a FAISS/HNSW production ANN index.
