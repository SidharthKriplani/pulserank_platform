# PulseRank Group 5 — Reranking Constraints

## Purpose

Group 5 turns raw model ranking into a governed marketplace ranking layer.

The reranker balances relevance against:

- seller exposure concentration
- category concentration
- intra-list diversity
- novelty
- catalog coverage

This is intentionally separated from the model. The model scores candidates; the reranker applies business and marketplace-health constraints before final display.

## Generated Local Data

    data/processed/reranked_lists.csv
    data/processed/rerank_audit_log.csv

## Evidence Artifacts

    outputs/evidence/diversity_guardrail_log.json
    outputs/evidence/diversity_report.json
    outputs/evidence/catalog_coverage_report.json
    outputs/reports/group5_reranking_constraints_report_v1.json
    outputs/validation/reranking_constraints_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/run_reranking_constraints_v1.py
    PYTHONPATH=. python3 scripts/validate_reranking_constraints_v1.py

## Claim Boundary

This is a deterministic reranking simulation. It is not a live marketplace policy engine and does not claim production seller fairness enforcement.
