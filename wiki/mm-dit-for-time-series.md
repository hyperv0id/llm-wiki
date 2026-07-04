---
title: "MM-DiT for Time Series"
type: concept
tags:
  - time-series
  - diffusion
  - transformer
  - multimodal
  - dual-stream
  - mm-dit
created: 2026-06-08
created: 2026-06-08
last_updated: 2026-07-04
source_count: 1
confidence: medium
status: active
---

# MM-DiT for Time Series

**MM-DiT for Time Series** is the architectural paradigm introduced by DiTS that adapts the Multimodal Diffusion Transformer (from text-to-image generation) to multivariate time series forecasting[^src-dits]. The core insight: **exogenous variates modulate the target variate analogously to how text prompts condition image generation** — they are not mere auxiliary features but a distinct modality stream deserving independent processing[^src-dits].

## The Core Insight

Traditional covariate-aware forecasters treat exogenous variates as auxiliary features that are either:
- **Concatenated with target tokens** (flattened time-variate grid → $O(T^2V^2)$ complexity, e.g., Timer-XL)[^src-dits].
- **Compressed into scalar modulation parameters** (AdaLN-only conditioning, e.g., TimeDiT, Sundial)[^src-dits].

DiTS's key observation is a **structural homogeneity** between multimodal generation (text → image) and covariate-aware forecasting (exogenous → endogenous)[^src-dits]. In both cases, one modality (text / covariates) provides conditional guidance that should influence the generation of another modality (image / target series) at fine granularity.

## How MM-DiT Adaptation Works

In the original MM-DiT (Esser et al., 2024), text and image latents flow as parallel streams through Transformer blocks, interacting exclusively through **joint attention** layers while maintaining separate FFNs for modality-specific refinement[^src-dits].

DiTS adapts this as follows[^src-dits]:

| Component | MM-DiT (Images) | DiTS (Time Series) |
|-----------|----------------|-------------------|
| Stream 1 | Text latent $c_{txt}$ | Exogenous variates $c$ |
| Stream 2 | Image latent $z$ | Endogenous variate $x$ |
| Shared module | Joint attention | Time Attention (temporal modeling) |
| Modality-specific | Separate FFNs | Separate QKV in Variate Attention |
| Conditioning | AdaLN from timestep + text pool | AdaLN from $Z_y$ = covariates + timestep |

Unlike MM-DiT which uses separate FFNs per stream, DiTS uses a **shared FFN** because time series patches across variates share a more stable latent manifold than heterogeneous image-text modalities[^src-dits].

## Why It Matters for Time Series

1. **Fine-grained conditional control**: Covariates participate in joint attention at every block, enabling token-level interaction rather than global scalar modulation[^src-dits].
2. **Low-rank decomposition**: The Time × Variate orthogonal factorization acts as a learnable low-rank decomposition of the full time-variate dependency matrix, avoiding $O(T^2V^2)$ complexity[^src-dits].
3. **Proven synergy**: Ablation confirms that the DiTS paradigm (Joint attention + AdaLN) outperforms Joint-only, Cross-only, and AdaLN-only alternatives[^src-dits].

## Related Concepts

- [[dual-stream-attention-time-series|Dual-Stream Attention]] — the specific attention mechanism implementing MM-DiT for time series
- [[dits|DiTS]] — the model that introduces this paradigm
- [[dit|DiT]] — the original Diffusion Transformer
- [[cora-tsfm|CoRA]] — alternative covariate-aware approach (post-head adaLN injection on frozen TSFMs)
- [[tsfm-covariate-adaptation-comparison]] — systematic comparison of covariate-aware adaptation methods
- [[flow-matching|Flow Matching]] — the generative framework used by DiTS

[^src-dits]: [[source-dits]]
