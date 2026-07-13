---
title: "QDF: Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models"
type: source-summary
tags:
  - time-series-forecasting
  - learning-objective
  - direct-forecast
  - label-autocorrelation
  - quadratic-form
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Summary

**QDF (Quadratic Direct Forecast)** is a model-agnostic training algorithm for multi-step time-series forecasting that learns a quadratic-form weighted objective with adaptive matrix $\Sigma$. By Hao Wang et al. (Xiaohongshu, PKU, ZJU et al.); arXiv:2511.00053 (ICLR 2026 preprint). Same author group as [[source-fredf|FreDF]] and [[source-distdf|DistDF]].[^src-qdf]

## Core Arguments

**1. Two gaps in MSE training.** Under Gaussian residuals, the conditional NLL is $L_\Sigma=\|Y-g_\theta(X)\|_{\Sigma^{-1}}^2$. MSE ($\Sigma=I$) ignores (i) [[label-autocorrelation|label autocorrelation]] (needs off-diagonals of $\Sigma^{-1}$) and (ii) [[heterogeneous-task-weights|heterogeneous task weights]] across horizons (needs non-uniform diagonals). True $\Sigma$ is unknown and hard to estimate from one $Y$ per $X$.[^src-qdf]

**2. FreDF / Time-o1 are incomplete.** Fourier ([[fredf|FreDF]]) and PCA (Time-o1) transforms only guarantee *marginal* decorrelation, not conditional independence, and still use equal component weights. On ECL ($T=96$), >61.4% of partial-correlation off-diagonals exceed 0.1; residual off-diagonals remain after those transforms, and conditional variances vary across steps.[^src-qdf]

**3. Learn proxy $\Sigma$, then train with $L_\Sigma$.** QDF solves a bilevel problem: inner loop fits $g_\theta$ on $D_{\mathrm{in}}$ with fixed $\Sigma$; outer loop updates $\Sigma$ for holdout generalization on $D_{\mathrm{out}}$ (meta-gradient through $\theta$). PSD via Cholesky $\Sigma=LL^\top$. Workflow: init $\Sigma=I$, refine over $K$ chronological train splits, then minimize $L_\Sigma$ on full train data. Training-only cost; inference unchanged.[^src-qdf]

## Experiments

On ETT×4, ECL, Weather, PEMS (input 96; horizons 96–720) with TQNet backbone, QDF leads averages vs TQNet/PDF/Fredformer/iTransformer/FreTS and other DF baselines (e.g., PEMS08 MSE 0.120 vs TQNet 0.139). Beats FreDF, Time-o1, Soft-DTW, Koopman, and DF as alternative objectives. Ablations: hetero-only and auto-corr-only each beat DF; full matrix is best. Gains transfer across TQNet, PDF, Fredformer, iTransformer. MAML-family optimizers for $\Sigma$ beat DF but trail QDF’s outer loop. Extra train cost stays <2 ms even at $T=720$.[^src-qdf]

## Limitations

Static learned $\Sigma$ is inflexible (hypernetwork objectives suggested). Extension beyond time series is open. Second-order Gaussian form may miss non-Gaussian residuals—orthogonal to DistDF’s OT route.[^src-qdf]

## Key Terminology

- **Quadratic-form weighted objective**: $L_\Sigma=\|Y-\hat Y\|_{\Sigma^{-1}}^2$ with learnable PSD $\Sigma$
- **Heterogeneous task weights**: non-uniform diagonals of $\Sigma^{-1}$
- **QDF**: bilevel $\Sigma$ learning + DF training under $L_\Sigma$

---

[^src-qdf]: [[source-qdf]]
