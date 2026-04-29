---
title: "Guided Layer Normalization (GLN)"
type: technique
tags:
  - normalization
  - conditional-modeling
  - traffic-forecasting
  - transformer
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Guided Layer Normalization (GLN)

**Guided Layer Normalization (GLN)** is a conditional normalization technique introduced in ConFormer (KDD 2026) that dynamically modulates internal representations based on traffic conditions. It addresses the limitation that standard LayerNorm uses fixed affine parameters (γ, β) shared across all inputs, which proves inadequate during accident scenarios where traffic patterns deviate significantly from normal conditions[^src-conformer].

## Mechanism

Given an input X₀, standard LayerNorm computes:

```
X̂ = γ (X₀ - μ) / σ + β
```

where μ and σ are the mean and standard deviation, and γ, β are learnable but fixed parameters.

GLN instead learns dynamic affine parameters from a contextual condition representation X_c:

```
γ, β = MLP(X_c)
GLN(X₀, γ, β) = γ · (X₀ - μ) / σ + β
```

The condition representation X_c is generated via graph propagation from the input embedding and accident/regulation features[^src-conformer].

## Interpretation

- **γ (scaling factor)**: Higher values increase sensitivity to abrupt changes (accidents), lower values suppress local fluctuations during normal traffic[^src-conformer].
- **β (shifting factor)**: Higher values emphasize node-specific features for accident response, lower values preserve global coherence[^src-conformer].
- **α (modulation factor)**: Scales residual connections, informed by curriculum learning principles to stabilize training[^src-conformer].

## Mathematical Reformulation

The self-attention with GLN can be reformulated as:

```
Softmax(√QK⊤/D_k) → Softmax(γ²Q'K'⊤ + γK'β⊤ + γQ'β⊤ + β²) / √D_k
```

where Q' = γ·(Q-μ_Q)/σ + β and similarly for K'[^src-conformer].

## Extensibility

GLN is not limited to ConFormer. Experiments show that adding GLN to GWNet achieves 2.63% MAE reduction across datasets, demonstrating its general applicability to other spatiotemporal models[^src-conformer].

## Related Techniques

- [[instance-normalization]] — RevIN strategy for handling distributional shifts

[^src-conformer]: [[source-conformer]]