---
title: "MMCKM — Micro-Macro Coupled Koopman Modeling on Graph for Traffic Flow Prediction"
type: source-summary
tags:
  - traffic-forecasting
  - koopman-operator
  - vehicle-trajectory
  - graph-pde
  - lagrangian-discretization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# MMCKM — Micro-Macro Coupled Koopman Modeling for Traffic Flow Prediction

ICLR 2026 Poster. Accepted paper.

## Problem

Multi-scale traffic flow modeling where individual vehicle dynamics (micro) and macroscopic traffic density evolution (macro) are jointly considered with bidirectional coupling — unlike traditional models that focus solely on vehicle behaviors or flow modeling in isolation[^src-mmckm].

## Core Contributions

1. **Vehicle-Centric PDE on Graphs**: Derives an advection–diffusion evolution equation discretized on a dynamic vehicle graph $G_t = (V_t, E_t, W_t)$, where vehicles are nodes connected by k-NN edges. Advection operator $C^{\text{adv}}$ is skew-symmetric (energy-preserving), and diffusion operator $L^{\text{diff}}$ is positive semi-definite (entropy-producing). This Lagrangian discretization preserves high-frequency vehicle-level perturbations that Eulerian grid-based methods inherently average away. Constructive parameterization via line-graph adjacency mask $M^{\text{loc}} \circ (P - P^\top)$ guarantees antisymmetry[^src-mmckm].

2. **Unified History-Free Koopman Modeling**: Both macro and micro dynamics are lifted to linear observation spaces via Koopman operators — time-invariant matrices $K_Z$ and $K_z$ that evolve from a single snapshot without requiring historical trajectories. The Markovian property eliminates the trajectory tracking and storage overhead of sequence-based methods. Spectral alignment loss $L_{\text{spec}}$ couples Koopman eigenvalues to graph-PDE spectra (diffusion $\leftrightarrow$ eigenvalue magnitude, advection $\leftrightarrow$ rotation frequency) for stability and interpretability[^src-mmckm].

3. **Physics-Guided Multi-Regime Micro Dynamics**: An Intent Discriminator (MoE) selects among 5 parameter-bounded Koopman evolvers — free flow, car-following, lane changing, merging, emergency — each with distinct spectral radius $\kappa_{\text{max}}$, rotation $\theta$, and actuation bounds $B_{\text{max}}$. A CrossAttention-based Koopman control path injects macro flow as control input $u_t$, with ISS (Input-to-State Stable) bounds guaranteeing no unbounded error growth over iterative applications[^src-mmckm].

## Key Results

On NGSIM (US-101 highway) and HighD (German highways), MMCKM achieves history-free trajectory prediction performance comparable to history-dependent SOTA methods (BAT, MS-STGCN, Vit-Traj) while outperforming the history-free baseline CV at all horizons (1–5s). The 0.1s operator excels at short-term (RMSE=0.33 at 1s), while the 1.0s operator achieves superior long-term accuracy via fewer iterative steps. Optimal operator interval on HighD is 0.4s (ADE=1.65). Ablation: removing diffusion term degrades macro prediction by 2.9–4.6% across horizons; Intent Discriminator contributes 29% improvement at 1s; Koopman control reduces error 37% at 5s. KDE bandwidth sensitivity shows optimum at 25m — too narrow (10m) injects noise that degrades diffusion operator learning below advection-only[^src-mmckm].

## Limitations & Future Work

KDE-estimated density lacks sensor-derived ground-truth benchmarks; cross-paper SOTA macro comparison deferred. Urban scenarios with heterogeneous graph structures remain unexplored. Future: learned edge weights as interpretable interaction measures for vehicle planning/control[^src-mmckm].

[^src-mmckm]: [[source-mmckm]]
