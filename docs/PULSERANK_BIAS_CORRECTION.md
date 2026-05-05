# PulseRank Group 4 — IPS Position-Bias Correction

## Purpose

Group 4 implements inverse propensity scoring for position-bias correction.

Logged clicks are biased because items displayed at higher ranks receive more examination probability. PulseRank estimates rank-level propensities from train-period impressions, clips low propensities, assigns IPS weights, and compares raw NDCG@10 against IPS-weighted NDCG@10.

## Generated Local Data

    data/processed/ips_training_examples.csv

## Evidence Artifacts

    outputs/evidence/propensity_by_rank_report.json
    outputs/evidence/bias_correction_report.json
    outputs/reports/group4_bias_correction_report_v1.json
    outputs/validation/bias_correction_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/run_bias_correction_v1.py
    PYTHONPATH=. python3 scripts/validate_bias_correction_v1.py

## Claim Boundary

This implementation uses rank-level empirical click propensities from the synthetic corpus. It is not a full cascade click model and does not claim live marketplace debiasing.
