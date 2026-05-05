# PulseRank Group 7 — Failure Scenarios and Demo Report

## Purpose

Group 7 creates the reviewer-facing defense layer for PulseRank.

It adds:

- scripted failure and recovery scenarios
- final demo report
- artifact inventory
- validation-chain check
- truth-boundary summary
- resume-safe project claim

## Evidence Artifacts

    outputs/evidence/failure_recovery_report.json
    outputs/reports/pulserank_demo_report.json
    outputs/reports/pulserank_demo_report.txt
    outputs/validation/demo_report_validation_v1.json

## Run

    PYTHONPATH=. python3 scripts/show_demo_report.py
    PYTHONPATH=. python3 scripts/validate_demo_report_v1.py

## Claim Boundary

This demo report summarizes local repo evidence. It does not claim production deployment, real users, real online experiments, or real revenue impact.
