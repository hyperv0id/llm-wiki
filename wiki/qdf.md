---
title: "QDF"
type: entity
tags:
  - time-series-forecasting
  - learning-objective
  - direct-forecast
  - plug-and-play
  - quadratic-form
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# QDF

**QDF (Quadratic Direct Forecast)** is a plug-and-play training algorithm that improves multi-step time-series forecasting by learning a quadratic-form weighted loss with an adaptive covariance-like matrix $\Sigma$. Proposed by Hao Wang et al. (Xiaohongshu / PKU / ZJU et al.), arXiv:2511.00053 (ICLR 2026 preprint). Overlapping author group with [[fredf|FreDF]] and [[source-distdf|DistDF]].[^src-qdf]

## Motivation

[[direct-forecast|Direct forecast (DF)]] models typically minimize step-wise MSE, which (i) ignores [[label-autocorrelation|label autocorrelation]] among future steps and (ii) assigns equal weight to every horizon step. Likelihood theory yields $L_\Sigma=\|Y-g_\theta(X)\|_{\Sigma^{-1}}^2$; off-diagonals of $\Sigma^{-1}$ capture residual dependence, diagonals encode [[heterogeneous-task-weights|heterogeneous task weights]]. Prior transforms ([[fredf|FreDF]], Time-o1) only guarantee *marginal* decorrelation and still use uniform component weights, leaving residual bias.[^src-qdf]

## Method

1. **Objective.** Train under the quadratic NLL form $L_\Sigma$ with PSD $\Sigma\in\mathbb{R}^{T\times T}$ (Cholesky reparameterization $\Sigma=LL^\top$).
2. **Bilevel $\Sigma$ learning.** Split train data into $D_{\mathrm{in}}/D_{\mathrm{out}}$ (and $K$ chronological folds). Inner loop: update $\theta$ on $D_{\mathrm{in}}$ with fixed $\Sigma$. Outer loop: meta-update $\Sigma$ through the effect on $\theta$ using holdout $L_\Sigma$ on $D_{\mathrm{out}}$.
3. **Final training.** Freeze (or use) the refined $\Sigma$ and minimize $L_\Sigma$ over the full training set. Inference uses the backbone only—no extra cost.[^src-qdf]

See [[quadratic-form-weighted-objective]] for the technique-level formulation and algorithms.

## Results Snapshot

- With TQNet backbone, best long-term averages on ETT/ECL/Weather/PEMS vs strong DF baselines.[^src-qdf]
- Beats FreDF, Time-o1, Soft-DTW, Koopman, and plain DF as alternative objectives on TQNet/PDF.[^src-qdf]
- Ablation: hetero-only (QDF†) and auto-corr-only (QDF‡) each help; both together best.[^src-qdf]
- Model-agnostic gains on TQNet, PDF, Fredformer, iTransformer; meta-learning variants for $\Sigma$ also beat DF.[^src-qdf]

## Relation to Sibling Work

| Method | Route | What it fixes |
|--------|-------|----------------|
| [[fredf\|FreDF]] | Frequency-domain likelihood alignment | Label autocorrelation via orthogonal bases (marginal) |
| [[source-distdf\|DistDF]] | Joint-distribution Wasserstein alignment | Avoids likelihood factorization / residual [[autocorrelation-bias]] |
| **QDF** | Learned quadratic $L_\Sigma$ | Autocorrelation *and* heterogeneous horizon weights under NLL form |

QDF stays in the likelihood family but *learns* $\Sigma$ for generalization, rather than fixing a Fourier/PCA transform or switching to OT.[^src-qdf]

## Links

- Source: [[source-qdf]]
- Concepts: [[label-autocorrelation]], [[heterogeneous-task-weights]], [[direct-forecast]], [[autocorrelation-bias]]
- Technique: [[quadratic-form-weighted-objective]]
- Related: [[fredf]], [[source-fredf]], [[source-distdf]], [[joint-distribution-wasserstein-alignment]], [[tqn|TQNet]], [[itransformer]]

---

[^src-qdf]: [[source-qdf]]
