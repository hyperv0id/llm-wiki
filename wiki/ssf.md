---
title: "SSF (Spectral Sheaf Filtering)"
type: entity
tags:
  - traffic-forecasting
  - spectral-methods
  - algebraic-topology
  - graph-neural-network
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# SSF (Spectral Sheaf Filtering)

SSF is a spatio-temporal modeling framework that bridges **cellular sheaf theory** (algebraic topology) with **spectral graph filtering** for traffic forecasting[^src-ssf]. It was submitted to ICLR 2026 and is under double-blind review.

## Architecture

SSF operates in four stages:

1. **Sheaf Construction**: Assigns $d$-dimensional stalk vector spaces to each node and learns linear **restriction maps** $F_{v \triangleleft e}$ per edge that encode how node features transform as information flows across the graph. These maps capture context-dependent, non-uniform interaction strengths — unlike standard GNNs where all edges propagate information uniformly[^src-ssf].

2. **Spectral Decomposition**: Computes eigendecomposition of the **[[sheaf-laplacian|sheaf Laplacian]]** $L_F = U\Lambda U^T$, generalizing graph Fourier analysis to the sheaf setting. Retains top-$k$ eigenmodes for compact, expressive representation[^src-ssf].

3. **Heat Kernel Filtering**: Transforms node features to the spectral domain ($\hat{X} = U^T X$), applies the heat kernel filter $\hat{g}_\text{heat} = \text{diag}(e^{-\alpha\lambda_1}, \ldots, e^{-\alpha\lambda_{Nd}})$, and inverse transforms back ($U \hat{X}_\text{filtered}$). This suppresses high-frequency noise while preserving low-frequency structural patterns[^src-ssf].

4. **Forecasting**: Stacks $L$ spectral filtering layers followed by an MLP prediction head to output multi-horizon forecasts[^src-ssf].

## Why Sheaves?

Standard GNNs propagate information uniformly along edges, assuming all connections carry equal weight. Real traffic networks exhibit **non-local, asymmetric** dependencies — an accident on one highway may cascade to distant neighborhoods while adjacent streets remain unaffected. Cellular sheaves address this by making information flow **edge-specific and learnable** via restriction maps, allowing the model to selectively route signals based on context[^src-ssf].

## Key Performance

| Dataset | 15min MAE | 30min MAE | 60min MAE |
|---------|-----------|-----------|-----------|
| METR-LA | 1.68 (SOTA) | 2.01 (SOTA) | 3.05 (SOTA) |
| PEMS-BAY | 0.85 (SOTA) | 1.29 (SOTA) | 1.77 (SOTA) |
| NAVER-Seoul | 3.41 (SOTA) | 3.58 (SOTA) | 3.84 (SOTA) |

SSF's error grows slowly with horizon, unlike baselines which degrade sharply at 60min[^src-ssf].

## Limitations

- Under review (not yet peer-reviewed). Anonymous code.
- Eigendecomposition complexity $O((Nd)^2 k)$ limits scale. Practical at $10^2\text{–}10^3$ nodes; larger graphs may need approximations[^src-ssf].

## See Also

- [[cellular-sheaf]] — the mathematical structure underlying SSF
- [[sheaf-laplacian]] — the generalized Laplacian used for spectral filtering
- [[over-smoothing-in-gnns]] — the problem SSF's sheaf structure mitigates
- [[traffic-forecasting]] — the application domain

[^src-ssf]: [[source-ssf]]
