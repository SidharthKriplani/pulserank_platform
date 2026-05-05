# PulseRank Platform

**Production-simulated marketplace recommendation and ranking decision system with IPS bias correction, delayed attribution, exposure governance, offline A/B simulation, and evidence artifacts.**

<p>
  <a href="./docs/index.html"><img alt="Dashboard" src="https://img.shields.io/badge/Live%20Dashboard-GitHub%20Pages-2ea44f?style=for-the-badge"></a>
  <img alt="Status" src="https://img.shields.io/badge/Status-PASS-22c55e?style=for-the-badge">
  <img alt="Decision" src="https://img.shields.io/badge/A%2FB%20Decision-HOLD_SIMULATED-f59e0b?style=for-the-badge">
  <img alt="IPS" src="https://img.shields.io/badge/IPS-Position%20Bias%20Correction-7c3aed?style=for-the-badge">
</p>

<p>
  <img alt="Display Rank" src="https://img.shields.io/badge/Display%20Rank%20Coverage-1.0-22c55e?style=flat-square">
  <img alt="Hybrid Recall" src="https://img.shields.io/badge/Hybrid%20Recall%40100-0.68986-0ea5e9?style=flat-square">
  <img alt="IPS NDCG" src="https://img.shields.io/badge/IPS%20NDCG%4010-0.52224-8b5cf6?style=flat-square">
  <img alt="Artifacts" src="https://img.shields.io/badge/JSON%20Artifacts-33-f97316?style=flat-square">
  <img alt="Failures" src="https://img.shields.io/badge/Failure%20Scenarios-15-ef4444?style=flat-square">
</p>

PulseRank is not a recommender notebook. It is a production-simulated ranking decision system that demonstrates how marketplace recommendations should be logged, evaluated, debiased, governed, and defended.

## Live Dashboard

Open locally:

    open outputs/dashboard/index.html

GitHub Pages source:

    Branch: main
    Folder: /docs

## Key Results

| Evidence | Result |
|---|---:|
| Sessions | 4132 |
| Items | 650 |
| Impressions | 41320 |
| Purchases | 539 |
| Display-rank coverage | 1.0 |
| Hybrid Recall@100 | 0.68986 |
| Holdout NDCG@10 | 0.13341 |
| IPS-weighted NDCG@10 | 0.52224 |
| After seller Gini@10 | 0.58228 |
| After catalog coverage@10 | 0.22615 |
| MetaSignal-compatible events | 15 |
| Failure scenarios | 15 |
| Offline A/B decision | HOLD_SIMULATED |

## Resume-Safe Claim

Built PulseRank, a production-simulated marketplace ranking system with display-rank impression logging, hybrid candidate generation, temporal holdout ranking evaluation, IPS position-bias correction, delayed conversion attribution, seller/category exposure governance, offline A/B simulation, failure recovery scenarios, and reproducible evidence artifacts.

## Claim Boundary

PulseRank is solo-built, non-production, and production-simulated. It does **not** claim real production deployment, real users, real traffic, real online A/B testing, real revenue optimization, real RL, real contextual bandits, or real ad-auction infrastructure.
