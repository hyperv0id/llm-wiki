---
title: "source-causalx"
type: source-summary
tags:
  - spatio-temporal
  - causal-inference
  - multimodal
  - diffusion
  - granger-causality
  - icml
  - pedestrian-trajectory
  - tropical-cyclone
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# CausalX: A Unified and Causally-Interpretable Plug-and-Play Model for Multi-modal Spatio-Temporal Forecasting

Zhang et al. (Zhejiang University of Technology, Shandong University, ICML 2026) propose **CausalX**, a unified plug-and-play model that learns dynamic causal-inspired graphs for multi-modal spatio-temporal forecasting, producing both improved accuracy and interpretable causal structures[^src-causalx]. Source: `raw/causalx-causally-interpretable-multimodal-st-forecasting.pdf`.

## Core Architecture: Two-Stage Pipeline

**Stage 1 — Multi-Source Causal Constraints Integration.** Given multi-modal inputs (1D scalar + 2D image-like), a GRU-CNN-GAT encoder constructs an initial fully-connected dynamic graph CG₀ whose edge weights serve as causal attribution scores[^src-causalx]. Four complementary causal analysis techniques supervise CG₀:

- **Granger causality**: variable-level predictive significance tests over max lag L, broadcast to edges[^src-causalx].
- **do-calculus**: intervenes on each node's feature via mean scaling, re-runs GAT, and measures embedding perturbation magnitude as interventional effect[^src-causalx].
- **TDMI** (Time-Delayed Mutual Information): max MI between environment variables and targets over delays τ ∈ [1, τmax][^src-causalx].
- **VAE**: latent encoding of each node; reconstructs node j's features from node i's latent; MSE is the generative dependency score[^src-causalx].

**Stage 2 — Diffusion-Based Graph Refinement.** CG₀ is treated as having residual uncertainty (no ground-truth causal labels available). A DDPM-style conditional diffusion process denoises the graph, guided by a binary prior graph Gprior encoding domain-consensus variable relations[^src-causalx].

## Plug-and-Play Integration

CausalX outputs both a graph-level representation g and a refined causal-inspired graph ĈG₀. Two integration modes: **Type A** (feature fusion) concatenates g with backbone features; **Type B** (graph replacement) substitutes ĈG₀ for the original adjacency[^src-causalx]. All auxiliary causal modules are disabled at inference — only the learned graph representation is fused.

## Key Results

Evaluated on two high-stakes domains: pedestrian trajectory (ETH/UCY, 3 backbones: MID, EigenTrajectory, SingularTrajectory) and tropical cyclone forecasting (2 backbones: TCNM, TC-Diffuser)[^src-causalx]. Consistent accuracy gains across all backbones and domains; e.g., TCNM 24h track error −19.9 km (−19.9%); MID AVG FDE −8.3%. Causal edge faithfulness validated via inference-time removal: top-edge removal causes the largest degradation; top-edge retention preserves most performance. Learned graphs exhibit strong stability (Spearman ρ = 0.87 ± 0.06 across seeds/splits) and physically plausible structures (e.g., recent positions dominate track prediction, intensity-pressure coupling, seasonal features for TC)[^src-causalx].

## Limitations

Requires end-to-end retraining (not zero-shot insertion). Gains on ZARA scenes are less stable due to unobserved environmental constraints (buildings, obstacles) not modeled by backbones[^src-causalx]. Training overhead from feature-level do-calculus, though offline Granger/TDMI precomputation and faster convergence partially offset this[^src-causalx].

[^src-causalx]: [[source-causalx]]
