---
title: "DiTS (Diffusion Transformers for Time Series)"
type: entity
tags:
  - time-series
  - diffusion
  - transformer
  - flow-matching
  - probabilistic-forecasting
  - arxiv-2026
  - tsinghua
created: 2026-06-08
created: 2026-06-08
last_updated: 2026-07-04
source_count: 1
confidence: medium
status: active
---

# DiTS (Diffusion Transformers for Time Series)

**DiTS** (Diffusion Transformers for Time Series) is a **flow-matching-based probabilistic time series forecasting model** proposed by Tsinghua University researchers (Zhang, Liu, Liu, Qiu, Wang, Wang & Long, arXiv Feb 2026)[^src-dits]. It adapts the Multimodal Diffusion Transformer (MM-DiT) architecture — originally designed for text-to-image generation — to multivariate time series by treating endogenous and exogenous variates as distinct modalities in a **dual-stream Transformer backbone**[^src-dits].

## Core Architecture

DiTS consists of $L$ stacked **DiTS Blocks**, each containing three sub-modules modulated by a global conditioning embedding $Z_y$[^src-dits]:

1. **Time Attention Module**: Shared self-attention over the temporal dimension for both $x$ and $c$ streams, capturing intra-variate temporal dynamics (akin to PatchTST's channel-independence strategy)[^src-dits].

2. **Variate Attention Module**: Joint attention where $x$ and $c$ project through independent QKV projections then attend jointly — modeling cross-variate (exogenous → endogenous) dependencies with per-stream output projections[^src-dits].

3. **Feed-Forward Network**: Shared FFN for intra-token refinement, maintaining stable latent manifolds across variates at the patch level[^src-dits].

Each sub-layer is modulated by AdaLN parameters $(\gamma, \beta)$ and a learnable gate $\alpha$ derived from $Z_y = \text{Mean}(\text{VariateEmbed}(c_i)) + \text{Sinusoidal}(t)$[^src-dits].

## Key Innovations

1. **MM-DiT for Time Series**: First framework to treat covariates as a distinct modality stream (not flattened into tokens or compressed into scalar modulation), enabling fine-grained token-level conditional control[^src-dits].
2. **Orthogonal dependency decomposition**: Explicitly factorizes modeling into temporal (intra-variate) and variate (inter-variate) axes — functioning as a learnable low-rank decomposition that avoids the $O(T^2V^2)$ complexity of full attention[^src-dits].
3. **Flow matching paradigm**: Replaces standard DDPM diffusion with rectified flow, enabling efficient 5-step sampling while maintaining probabilistic output quality[^src-dits].

## Position in DiT-for-TS Landscape

DiTS differs from prior DiT-based time series models[^src-dits]:
- **TimeDiT** (KDD 2025): Uses single-stream DiT with AdaLN-only conditioning — no explicit covariate stream.
- **LDT** (AAAI 2024): Latent diffusion transformer with single-stream structure, no sophisticated exogenous variate fusion.
- **Sundial** (2025): Asymmetric encoder-denoiser design (MAR-style), not conventional DiT architecture.
- **CoRA** (ICLR 2026): Freezes pre-trained TSFMs (including Sundial) and injects covariates post-hoc via adaLN — an orthogonal adaptation paradigm that does not require building a new backbone.

## Performance

- **FEV-Bench**: #1 in avg WQL (0.070) and MASE (0.601), beating Chronos-2, TabPFN-TS, Moirai-2.0, Sundial-Base, etc.[^src-dits]
- **EPF**: Avg MSE 0.274, 10%+ improvement over TimeXer; especially dominant at 360-step horizon[^src-dits]
- **Univariate LTSF**: Beats PatchTST on all 6 benchmarks[^src-dits]

## Limitations & Status

- arXiv preprint (Feb 2026), not yet peer-reviewed[^src-dits].
- Code not public[^src-dits].
- Single RTX 4090 experiments; scaling behavior unexplored[^src-dits].
- Only univariate target evaluated on FEV-Bench[^src-dits].

[^src-dits]: [[source-dits]]

## Related Pages

- [[mm-dit-for-time-series]] — MM-DiT paradigm for time series
- [[cora-tsfm|CoRA]] — alternative covariate-aware approach (freeze backbone + post-head injection)
- [[tsfm-covariate-adaptation-comparison]] — systematic comparison of adaptation methods
- [[dual-stream-attention-time-series]] — the attention mechanism behind DiTS
- [[sundial]] — backbone reused by CoRA for covariate-aware forecasting
