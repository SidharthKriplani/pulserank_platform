# PulseRank Scope

PulseRank is a production-simulated marketplace recommendation and ranking decision system.

It is not a recommender notebook. The purpose is to demonstrate ranking-system judgment:

- position-logged impressions
- IPS bias correction
- delayed conversion attribution
- seller and category exposure governance
- exploration logging
- offline evaluation
- A/B simulation
- reproducible evidence artifacts

## Core Build Spine

1. Deterministic corpus generation.
2. Impression log with display_rank on every impression.
3. Candidate generation baseline: popularity plus content similarity.
4. Ranking baseline with temporal train/test split.
5. IPS position-bias correction with propensity clipping.
6. Reranking constraints: MMR, seller Gini, category exposure, novelty.
7. Delayed conversion attribution across 1/3/7-day windows.
8. Offline evaluation harness.
9. Offline A/B simulation.
10. Failure and recovery scenarios.
11. show_demo_report.py evidence runner.
12. Static dashboard and GitHub showcase.

## Truth Boundary

PulseRank is solo-built, non-production, and production-simulated.

It does not claim real production deployment, real users, real traffic, real online experiments, real revenue optimization, contextual bandit/RL optimization, or ad-auction infrastructure.
