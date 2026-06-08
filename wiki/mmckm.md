---
title: "MMCKM"
type: entity
tags:
  - traffic-forecasting
  - koopman-operator
  - vehicle-trajectory-prediction
  - graph-pde
  - iclr
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# MMCKM — Micro-Macro Coupled Koopman Modeling

MMCKM is a history-free traffic prediction framework (ICLR 2026 Poster) that unifies **microscopic vehicle trajectory prediction** and **macroscopic traffic density evolution** within a single Koopman operator-based architecture on dynamic vehicle graphs[^src-mmckm].

## Core Architecture

The framework models traffic as a dynamic weighted directed graph $G_t = (V_t, E_t, W_t)$ where nodes are vehicles, edges are k-NN connections (k=6), and edge weights encode advection and diffusion coefficients learned via GNNs[^src-mmckm].

### Macro: Vehicle-Centric Graph PDE

MMCKM discretizes the LWR traffic flow PDE (advection + diffusion) directly onto vehicles as Lagrangian graph nodes, preserving high-frequency perturbations that grid-based methods inherently lose. The evolution is $\dot{\rho} = -C^{\text{adv}}\rho + L^{\text{diff}}\rho$, where $C^{\text{adv}}$ is skew-symmetric (energy-preserving advection) and $L^{\text{diff}}$ is PSD (entropy-producing diffusion)[^src-mmckm]. Density is then lifted to a Koopman observation space $Z_t = \phi_Z(G_t)$, evolved linearly via $Z_{t+1} = K_Z Z_t$, and decoded to density predictions[^src-mmckm]. A commutator regularization $L_{\text{JAD}} = \|L^{\text{diff}}C^{\text{adv}} - C^{\text{adv}}L^{\text{diff}}\|_F^2$ reduces basis rotation, and spectral alignment $L_{\text{spec}}$ couples Koopman eigenvalues with PDE operator spectra[^src-mmckm].

### Micro: Multi-Regime Koopman with Flow Control

Vehicle states $x_t^e$ are lifted to observation space $z_t = \phi_z(x_t^e)$, evolved as $z_{t+1} = K_z z_t + B_z u_t$, where the control input $u_t = \text{CA}(z_t, Z_t)$ is a CrossAttention block injecting macroscopic flow influence[^src-mmckm]. An Intent Discriminator (MoE) selects among 5 parameter-bounded Koopman operators — each with distinct spectral radius $\kappa_{\text{max}}$, rotation $\theta$, and actuation bound $B_{\text{max}}$ — mapped to driving modes: free flow, car-following, lane changing, merging, emergency[^src-mmckm]. ISS-style bounds ($\kappa(K_z) < 1$, bounded $u_t$ via Sigmoid) guarantee that errors decay geometrically without unbounded growth[^src-mmckm].

## Key Properties

- **History-Free (Markovian)**: Prediction uses only the current snapshot $G_t$, eliminating trajectory tracking and storage overhead of sequence-based methods[^src-mmckm].
- **Bidirectional Coupling**: Macro flow explicitly influences micro vehicle dynamics via Koopman control, and vehicle perturbations propagate into flow via diffusion term[^src-mmckm].
- **Interpretable Edge Weights**: Learned advection ($W^{\text{adv}}$) and diffusion ($W^{\text{diff}}$) weights quantify vehicle-to-vehicle interaction intensities, a capability unique to vehicle-centric formulations[^src-mmckm].
- **Linear Complexity**: Inference dominated by sparse GNN message passing $O(kNd)$ + Koopman evolution $O(T d^2)$, vs $O(T(Nd^2 + kNd))$ for spatiotemporal GNNs[^src-mmckm].

## Results Summary

SOTA history-free trajectory prediction on NGSIM and HighD, matching history-dependent methods (BAT, MS-STGCN, Vit-Traj) while outperforming CV at all horizons. Optimal operator interval of 0.4s on HighD. Ablation confirms all three modules essential[^src-mmckm].

## Related Concepts

- [[vehicle-centric-graph-traffic-pde]] — Lagrangian PDE discretization on vehicle graphs
- [[micro-macro-coupled-koopman-modeling]] — unified Koopman framework
- [[intent-discriminator-koopman]] — MoE-based driving mode selection

[^src-mmckm]: [[source-mmckm]]
