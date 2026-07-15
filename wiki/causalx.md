---
title: "CausalX"
type: entity
tags:
  - spatio-temporal
  - causal-inference
  - multimodal
  - diffusion
  - granger-causality
  - plug-and-play
  - icml
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# CausalX

**CausalX** is a unified, causally interpretable, plug-and-play model for multi-modal spatio-temporal forecasting, proposed by Zhang et al. (Zhejiang University of Technology, ICML 2026)[^src-causalx]. It constructs dynamic causal-inspired graphs across modalities and time steps, providing structured explanations of which variables and time lags contribute most to each prediction outcome.

## Core Idea

Rather than fixing a predefined causal structure, CausalX learns instance-specific dynamic graphs whose edge weights represent causal attribution strength[^src-causalx]. The key insight: since ground-truth causal structures are unavailable in real-world multi-modal forecasting, CausalX aggregates multiple complementary causal signals — predictive (Granger), interventional (do-calculus), temporal (TDMI), and generative (VAE) — as joint supervision. The residual uncertainty is then modeled through a diffusion-based denoising process guided by domain prior graphs[^src-causalx].

## Architecture

CausalX consists of two stages[^src-causalx]:

1. **Multi-Source Causal Constraints Integration**: A GRU-CNN-GAT encoder builds an initial fully-connected dynamic graph from multi-modal inputs (1D scalar variables + 2D image-like fields). Four causal analysis techniques — [[granger-causality|Granger causality]], [[do-calculus]], TDMI, and VAE reconstruction — produce supervision graphs that jointly constrain the learned attention weights from predictive, interventional, temporal, and generative perspectives.

2. **Diffusion-Based Graph Refinement**: The initial graph CG₀ is treated as having unexplained residual components. A DDPM-style conditional diffusion process iteratively denoises the graph, conditioned on node features, a domain prior graph Gprior, and timestep t, to recover missing causal relations.

## Plug-and-Play Design

CausalX outputs (i) a graph-level representation g and (ii) a refined causal-inspired graph. Two integration modes support diverse backbones[^src-causalx]:

- **Type A (Feature Fusion)**: Concatenate g with backbone features, project, then feed to decoder (e.g., TCNM, MID).
- **Type B (Graph Replacement)**: Replace or augment backbone adjacency with the causal graph (e.g., EigenTrajectory).

At inference, all auxiliary causal modules are disabled — CausalX reduces to a single learned causal graph representation, with negligible latency overhead.

## Key Properties

- **Architecture-agnostic**: Successfully integrated into 5 distinct backbones spanning two domains (pedestrian trajectory: MID, EigenTrajectory, SingularTrajectory; TC forecasting: TCNM, TC-Diffuser)[^src-causalx].
- **Consistent accuracy gains**: e.g., TCNM: −19.9 km at 24h (−19.9% track error); MID: AVG FDE −8.3%[^src-causalx].
- **Interpretable**: Chord diagrams and per-constraint visualizations reveal physically plausible causal structures (recent position → track, intensity-pressure coupling, seasonal features for TC)[^src-causalx].
- **Stable**: Graphs show strong consistency across random seeds and data splits (Spearman ρ = 0.87 ± 0.06)[^src-causalx].
- **Causally faithful**: Removing top-ranked edges causes the largest prediction degradation; retaining only top-ranked edges preserves most performance[^src-causalx].

## Comparison with Related Causal ST Models

| Model | Causal Approach | Venue |
|-------|----------------|-------|
| [[e2-cstp\|E²-CSTP]] | DeepSHAP + back-door adjustment; dual-branch causal inference | NeurIPS 2025 |
| [[source-cast\|CaST]] | SCM + back-door (temporal OoD) + front-door (spatial) adjustment | NeurIPS 2023 |
| **CausalX** | Multi-source causal constraints (Granger/do/TDMI/VAE) + diffusion graph refinement | ICML 2026 |

CausalX differs from E²-CSTP and CaST in that it does not rely on SCM-based confounding adjustment; instead, it treats the graph itself as the object of causal learning, using multiple causal analysis signals as joint supervision without requiring a pre-specified causal DAG[^src-causalx].

## Links

- [[source-causalx]] — source summary
- [[granger-causality]] — Granger causality
- [[multi-source-causal-constraints]] — multi-source causal constraints concept
- [[causal-time-series-forecasting]] — causal TS forecasting paradigm
- [[diffusion-model]] — diffusion models
- [[e2-cstp]] — related causal multi-modal ST model

[^src-causalx]: [[source-causalx]]
