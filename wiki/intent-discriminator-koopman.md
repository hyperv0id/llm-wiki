---
title: "Intent Discriminator for Koopman Operators"
type: technique
tags:
  - koopman-operator
  - mixture-of-experts
  - driving-mode-classification
  - trajectory-prediction
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Intent Discriminator for Koopman Operators

A mixture-of-experts (MoE) gating mechanism introduced by [[mmckm|MMCKM]] (ICLR 2026) that selects among a family of parameter-bounded Koopman operators, each specialized for a distinct driving regime[^src-mmckm].

## Motivation

Vehicle microscopic dynamics exhibit varying features across driving scenarios — free flow, car-following, lane changing, merging, and emergency maneuvers. A single Koopman operator must account for these fundamentally different dynamical regimes, but directly increasing the operator dimension makes eigendecomposition computationally prohibitive. Different modes also present distinct Koopman spectra and control responses[^src-mmckm]:

- **Free flow**: Near-constant maximum speed → $\kappa(K_z) \approx 1$ (slow decay), small oscillation (imaginary eigenvalues near 0), weak external influence
- **Car-following**: Longitudinal coupling to lead vehicle → stronger actuation $B_{\text{max}} > 0$
- **Lane changing**: Coupled longitudinal + lateral oscillation → explicit vibration period ($\theta_{\text{mean}} \neq 0$)
- **Merging**: Lateral entry + acceleration → distinct from lane changing ($\theta_{\text{mean}}$ initialized negative)
- **Emergency**: Rapid acceleration/braking → $\kappa_{\text{max}} < 1$ (fast decay), larger external influence

## Design

### Parameter-Bounded Koopman Operator Family

Each Koopman operator $K_z$ is constructed from $N^c$ $2\times2$ complex blocks and $N^r$ $1\times1$ real blocks[^src-mmckm]:

$$K_c = R \times \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}, \quad R = \kappa_{\text{max}} \cdot \sigma(\eta)$$

where $\eta$ is the advection eigenvalue (governs flow propagation speed), $\theta_{\text{mean}}$ and $\theta_{\text{std}}$ represent oscillation frequency from diffusion eigenvalues, and $\kappa_{\text{max}}$ bounds the spectral radius. Real blocks apply the same $R$ constraint[^src-mmckm].

The 5 operators differ by hyperparameters[^src-mmckm]:

| Mode | $\kappa_{\text{max}}$ | $B_{\text{max}}$ | $\theta_{\text{std}}$ | $\theta_{\text{mean}}$ | Physical interpretation |
|------|----------------------|-------------------|----------------------|----------------------|------------------------|
| Free flow | 0.95 | 0.20 | 0.01 | 0.00 | Persist speed, minimal interaction |
| Car-following | 0.85 | 0.60 | 0.02 | 0.00 | Lead vehicle governs behavior |
| Lane changing | 0.90 | 0.75 | 0.08 | 0.25 | Lateral + longitudinal oscillation |
| Merging | 0.88 | 0.80 | 0.05 | -0.15 | Entry + acceleration |
| Emergency | 0.70 | 0.40 | 0.01 | 0.00 | Rapid state change |

### Intent Discriminator (MoE Gating)

Implemented as an MLP that evaluates the current vehicle state $x_t^e$ and the graph-based macroscopic observation $Z_t$ to select the most consistent Koopman operator from the candidate set[^src-mmckm]. Training labels are generated via data preprocessing calibrated by acceleration and lane variance — fully reproducible without learned classifiers or manual annotation[^src-mmckm].

## Ablation Results

On HighD at 0.2s operator interval, removing the Intent Discriminator (MMCKM-I) severely degrades short-term prediction: RMSE increases from 0.29 to 0.74 at 1s (29% contribution). However, its effectiveness diminishes at longer horizons (RMSE at 5s: 2.73 vs 3.81 — only ~28% relative), as the latent macroscopic state $Z_t$ evolves independently and accumulates errors degrading intent classification accuracy. Maintaining accurate Intent Discrimination over long horizons would require simultaneous state updates for all surrounding vehicles — computationally prohibitive[^src-mmckm].

## Significance

This technique avoids the high computational cost of a single over-generalized Koopman operator by leveraging a structured ensemble of specialized operators. The Intent Discriminator serves as a lightweight gating mechanism that adaptively aligns operator selection with underlying driving intent, enabling efficient multi-regime dynamics modeling without eigendecomposition at inference time[^src-mmckm].

[^src-mmckm]: [[source-mmckm]]
