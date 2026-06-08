---
title: "Dual-Stream Attention for Time Series"
type: technique
tags:
  - time-series
  - attention
  - transformer
  - dual-stream
  - variate-attention
  - temporal-attention
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Dual-Stream Attention for Time Series

**Dual-Stream Attention** is the core architectural mechanism of DiTS that decomposes multivariate time series modeling into orthogonal axes: **Time Attention** for intra-variate temporal dynamics and **Variate Attention** for inter-variate (cross-covariate) dependencies[^src-dits]. This design functions as a learnable low-rank decomposition, avoiding the prohibitive $O(T^2V^2)$ cost of full attention over the flattened time-variate grid[^src-dits].

## Design Rationale

Multivariate time series exhibit two fundamentally different types of dependency[^src-dits]:

1. **Intra-variate (temporal)**: How a single variate evolves over time — autocorrelation, trends, seasonality. These follow consistent temporal patterns that generalize across variates[^src-dits].
2. **Inter-variate (cross-variate)**: How different variates influence each other — e.g., grid load affects electricity prices. These dependencies are typically **low-rank** and benefit from joint modeling[^src-dits].

Existing architectures typically model only one dimension well[^src-dits]:
- **PatchTST/iTransformer**: Channel-independent → strong temporal modeling, zero variate interaction.
- **Timer-XL**: Full attention → captures both but at $O(T^2V^2)$ cost.
- **TimeXer**: Asymmetric — $O(T^2)$ temporal + $O(V)$ cross-attention, but variate interaction is coarse.

DiTS's dual-stream design provides **fine-grained variate interaction at low cost** ($O(T^2 + V)$) by separating the two dependency types into dedicated modules[^src-dits].

## Components

### Time Attention Module

```
H_time_s = MSA_time(Mod(Z_s, γ_T, β_T))
Ẑ_s = Z_s + α_T ⊙ H_time_s    for s ∈ {x, c}
```

- **Shared** across $x$ and $c$ streams (temporal patterns generalize across variates)[^src-dits].
- Self-attention over the **temporal dimension** $T$ (patch tokens)[^src-dits].
- Follows PatchTST's proven efficacy in temporal extrapolation[^src-dits].
- Complexity: $O(T^2)$ per stream[^src-dits].

### Variate Attention Module

```
U_s = Mod(Ẑ_s, γ_V, β_V)  for s ∈ {x, c}
[H_var_x; H_var_c] = Joint-Attn({Q_s, K_s, V_s, W_s^O}_s)
Z̃_s = Ẑ_s + α_V ⊙ H_var_s
```

- **Independent QKV projections** per stream (respecting heterogeneous latent spaces of endogenous vs exogenous)[^src-dits].
- **Joint attention**: $x$ and $c$ tokens attend jointly in the variate dimension, enabling fine-grained exogenous → endogenous information flow[^src-dits].
- **Per-stream output projections** $W_s^O$ for modality-specific reconstruction[^src-dits].
- Complexity: $O(V)$ where $V$ is the number of variates[^src-dits].

### Feed-Forward Network

```
Z_s = Z̃_s + α_F ⊙ FFN(Mod(Z̃_s, γ_F, β_F))
```

- **Shared** FFN across streams — time series patch manifolds are more stable across variates than image-text modalities[^src-dits].
- Point-wise operation for intra-token refinement[^src-dits].

### Adaptive Modulation

All three sub-modules are modulated by $Z_y$ (covariate mean pooling + sinusoidal timestep embedding) via AdaLN[^src-dits]:
```
{α_m, β_m, γ_m}_{m∈{T,V,F}} = MLP_mod(Z_y)
Mod(h, γ, β) = γ ⊙ LN(h) + β
```

The learnable gate $\alpha$ controls the residual contribution of each sub-module, enabling the model to dynamically balance temporal vs variate processing[^src-dits].

## Complexity Comparison

| Architecture | Attention Complexity | Variate Modeling |
|-------------|---------------------|-----------------|
| PatchTST (CI) | $O(T^2)$ | None |
| iTransformer | $O(V^2)$ | Full (variate-as-token) |
| Timer-XL | $O(T^2V^2)$ | Full (flattened grid) |
| TimeXer | $O(T^2 + V)$ | Coarse (cross-attention) |
| **DiTS** | $O(T^2 + V)$ | **Fine-grained (joint attention)** |

DiTS matches TimeXer's asymptotic complexity while providing richer variate interaction through joint (rather than cross) attention[^src-dits].

## Ablation Validation

On the EPF dataset (L=168, H=24)[^src-dits]:
- DiTS (Time + Variate attention) > TimeXer-style ~ Prefix-style > Timer-XL-style ~ iTransformer-style.
- Removing variate attention (flattening all tokens) causes the largest degradation, confirming that the dual-stream factorization is the key architectural advantage.

[^src-dits]: [[source-dits]]
