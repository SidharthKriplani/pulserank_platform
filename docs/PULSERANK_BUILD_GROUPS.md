# PulseRank Build Groups

## Group 0 — Repo Foundation

Create repo scaffold, config, docs, validation, README, and truth boundary.

## Group 1 — Corpus + Schemas

Generate 90-day deterministic marketplace corpus:

- user_sessions
- item_catalog
- impression_log with display_rank
- purchase_log
- policy_config snapshot
- corpus_summary.json

## Group 2 — Candidate Generation

Implement popularity and content-based candidate generators.

Artifacts:

- candidate_generation_report.json
- candidate_recall_report.json

## Group 3 — Ranking Baseline

Implement temporal train/test split and baseline ranker.

Artifacts:

- ranking_baseline_report.json
- offline_eval_log.json
- model_registry.json

## Group 4 — IPS Bias Correction

Implement propensity estimation, IPS weights, clipping, and before/after evaluation.

Artifacts:

- bias_correction_report.json
- propensity_by_rank_report.json

## Group 5 — Reranking Constraints

Implement MMR, seller Gini, category exposure, novelty, and coverage constraints.

Artifacts:

- diversity_guardrail_log.json
- diversity_report.json
- catalog_coverage_report.json

## Group 6 — Attribution + A/B Simulation

Implement delayed conversion attribution and offline A/B simulation.

Artifacts:

- attributed_label_log.json
- conversion_attribution_report.json
- ab_simulation_results.json
- metasignal_integration_events.json

## Group 7 — Failure Scenarios + Demo Report

Implement failure/recovery scenarios and demo runner.

Artifacts:

- failure_recovery_report.json
- pulserank_demo_report.json
- validation summary

## Group 8 — GitHub Showcase

Add README polish, GitHub Pages dashboard, release notes, and defense document.
