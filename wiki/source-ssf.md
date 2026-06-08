---
title: "SSF: Spectral Sheaf Filtering — A Topological Approach to Spatio-Temporal Modeling"
type: source-summary
tags:
  - traffic-forecasting
  - spectral-methods
  - algebraic-topology
  - cellular-sheaf
  - graph-neural-network
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# SSF: Spectral Sheaf Filtering (ICLR 2026, under review)

**Anonymous authors. Under double-blind review at ICLR 2026.** Code: [github.com/anonymous-submisssion/SSF](https://github.com/anonymous-submisssion/SSF).

## Summary

SSF (Spectral Sheaf Filtering) is the first framework to model spatio-temporal data using **cellular sheaves** from algebraic topology. It redefines information propagation on graphs by assigning vector spaces (stalks) to nodes and edges, along with **restriction maps** that learn context-dependent, locally adaptive transformations between them. SSF then applies **spectral filtering over the sheaf Laplacian**, using a heat kernel filter to decompose and modulate frequency components of graph signals, effectively mitigating [[over-smoothing-in-gnns|oversmoothing]].

## Core Contributions

1. **Cellular Sheaf Construction**: Assigns $d$-dimensional stalk vector spaces to nodes and learns linear restriction maps $F_{v \triangleleft e}$ per edge, encoding how features transform differently across graph regions. The **sheaf Laplacian** $L_F$ aggregates these transformations, generalizing the combinatorial graph Laplacian ($L_F$ reduces to it when $d=1$ and maps are identity)[^src-ssf].

2. **Sheaf Fourier Analysis**: Proves the sheaf Laplacian admits eigendecomposition $L_F = U\Lambda U^T$ (Theorem 1). Small eigenvalues correspond to low-frequency (globally smooth) components; large eigenvalues capture high-frequency (localized, oscillatory) patterns. This extends classical spectral graph theory to the sheaf setting[^src-ssf].

3. **Heat Kernel Spectral Filter**: Applies $g_\text{heat}(\lambda) = e^{-\alpha\lambda}$ in the spectral domain — transform $X$ via $U^T$, filter with $\hat{g}_\text{heat}$, then inverse transform. This selectively suppresses high-frequency noise and emphasizes coherent low-frequency components[^src-ssf].

## Results

Evaluated on 5 benchmarks (METR-LA, PEMS-BAY, PEMS04, PEMS08, NAVER-Seoul) across horizons 3/6/12 (15/30/60 min). SSF achieves SOTA on all metrics, with particularly dramatic gains in long-horizon forecasting and on the challenging NAVER-Seoul dataset (MAPE 1.03% at 15min vs. best baseline 8.32%). Key findings:

- In METR-LA: MAE 1.68 (vs. ModWaveMLP 2.20) at 15min; MAE 3.05 (vs. 3.40) at 60min. Error growth from 15→60 min is substantially flatter than baselines[^src-ssf].
- Ablation confirms spectral filtering is critical — removing it causes severe degradation on NAVER-Seoul (RMSE 20.72 vs. 3.89 at 60min). Small $k=3$ eigenvalues optimal; larger $k$ introduces noise[^src-ssf].
- Stalk dimension $d$: higher $d$ reduces error at the cost of computation. Spectral filter reduces iteration time across all $d$ values by offsetting eigendecomposition overhead[^src-ssf].

## Limitations

- Under review — not yet peer-reviewed. Code is anonymous. Confidence: **medium**.
- Eigendecomposition is $O((Nd)^2 k)$, quadratic in node count; practical at $N \sim 10^2\text{–}10^3$ but may challenge larger graphs. Forward pass at $N=1500, d=6, k=5$ is 0.34s[^src-ssf].

[^src-ssf]: [[source-ssf]]
