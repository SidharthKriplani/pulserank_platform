# PulseRank Platform

**Production-simulated marketplace recommendation and ranking decision system.**

<p>
  <img alt="Status" src="https://img.shields.io/badge/Status-Active%20Build-2ea44f?style=for-the-badge">
  <img alt="Ranking" src="https://img.shields.io/badge/Ranking-Marketplace%20Decision%20System-2563eb?style=for-the-badge">
  <img alt="IPS" src="https://img.shields.io/badge/IPS-Position%20Bias%20Correction-7c3aed?style=for-the-badge">
  <img alt="Attribution" src="https://img.shields.io/badge/Attribution-Delayed%20Conversion-f59e0b?style=for-the-badge">
</p>

PulseRank is a solo-built, non-production, production-simulated ranking system for marketplace recommendation workflows.

It is designed to demonstrate ranking-system judgment beyond a recommender notebook:

- display-rank impression logging
- candidate generation
- ranking baseline
- IPS position-bias correction
- propensity clipping
- MMR reranking
- seller/category exposure governance
- cold-start and exploration policy
- delayed conversion attribution
- offline evaluation harness
- offline A/B simulation
- failure and recovery scenarios
- evidence artifacts and demo report

## At a Glance

| Layer | What it proves |
|---|---|
| Impression logging | Every recommendation impression must carry display_rank |
| IPS correction | Logged clicks are position-confounded and must be debiased |
| Delayed attribution | CTR is not enough; purchases arrive after impressions |
| Reranking constraints | Final ranking is not raw model top-K |
| Seller exposure | Marketplace health requires exposure governance |
| Offline evaluation | Ranking requires relevance, diversity, coverage, business, and bias metrics |
| A/B simulation | Ranking changes must connect to CVR, revenue/session, and guardrails |
| Evidence artifacts | Every claim must be backed by generated files |

## Run Validation

    PYTHONPATH=. python3 scripts/validate_scaffold_v1.py

## Claim Boundary

PulseRank is solo-built, non-production, and production-simulated.

It does not claim:

- real production deployment
- real users served
- real production traffic
- real online A/B test
- real revenue optimization
- real RL or contextual bandit optimization
- real ad auction infrastructure

## Status

Group 0 foundation is active. Core implementation begins with corpus generation and impression logging.
