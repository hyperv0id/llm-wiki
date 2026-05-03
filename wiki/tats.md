---
title: "TaTS (Texts as Time Series)"
type: entity
tags:
  - multimodal-time-series
  - forecasting
  - imputation
  - plug-and-play
  - text-alignment
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# TaTS (Texts as Time Series)

**TaTS** (Texts as Time Series) is a plug-and-play multimodal time series framework proposed by Li et al. from UIUC, Meta, and IBM Research, accepted at ICLR 2026[^src-language-in-the-flow-of-time]. It transforms time-series-paired texts into auxiliary variables that augment the original numerical time series, enabling any existing time series model to handle multimodal data without architecture modification.

## Core Insight

Motivated by the **Platonic Representation Hypothesis (PRH)** and the discovery of **[[chronological-textual-resonance|Chronological Textual Resonance (CTR)]]** — the observation that time-series-paired texts exhibit periodic properties mirroring the original time series — TaTS treats paired texts as **auxiliary variables** of the time series[^src-language-in-the-flow-of-time]. This is analogous to how different variables in a multivariate time series share similar periodicity properties.

## Architecture

TaTS consists of three simple steps:

### 1. Text Encoding

Paired texts $\mathbf{S} = \{s_1, s_2, \dots, s_T\}$ are encoded using a pre-trained LLM (GPT-2 by default):

$$\mathbf{e}_t = \mathcal{H}_{\text{text}}(s_t) \in \mathbb{R}^{d_{\text{text}}}$$

### 2. Dimensionality Reduction

Since $d_{\text{text}}$ is typically much larger than the number of time series variables, an MLP reduces dimensionality:

$$\mathbf{z}_t = \text{MLP}(\mathbf{e}_t) \in \mathbb{R}^{d_{\text{mapped}}}$$

### 3. Concatenation as Auxiliary Variables

The mapped embeddings are concatenated with the original time series to form a unified multimodal sequence:

$$\mathbf{U} = [\mathbf{X}; \mathbf{Z}^{\intercal}] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$$

This augmented sequence is fed into any existing time series model $\mathcal{F}$:

$$\widehat{\mathbf{X}}_{T+1:T+H} = \mathcal{F}(\mathbf{U}_{1:T})[:N]$$

The mapping MLP and time series model are jointly trained using MSE loss[^src-language-in-the-flow-of-time].

## Key Properties

| Property | Description |
|----------|-------------|
| **Plug-and-play** | No modification to existing model architectures |
| **Model-agnostic** | Compatible with Transformer-based, linear, and frequency-based models |
| **Task-agnostic** | Works for both forecasting and imputation |
| **CTR-dependent** | Performance gains correlate with CTR level (lower TT-Wasserstein → larger improvement) |
| **Encoder-flexible** | Works with various text encoders (GPT-2, BERT, etc.) |

## Performance

TaTS was evaluated on 18 datasets across 9 time series models:

- **Forecasting**: Average >5% improvement on 6/9 datasets; >30% on Environment dataset
- **Imputation**: Up to 30% improvement over baselines
- **Consistency**: Gains across both short-term ({6,8,10,12}) and long-term ({48,96,192,336}) prediction horizons[^src-language-in-the-flow-of-time]

## Comparison with Related Models

### vs. VoT

| Dimension | TaTS | [[vot|VoT]] |
|-----------|------|-----|
| Approach | Text as auxiliary variables | LLM reasoning + multi-level alignment |
| LLM usage | Feature extraction only | Feature extraction + reasoning |
| Architecture | Plug-in (no modification) | Dual-branch (event-driven + numerical) |
| Text types | Time-paired texts | Exogenous + Endogenous |
| Fusion | Concatenation | ETA (representation) + AFF (prediction) |
| Complexity | Low (MLP + concat) | High (LLM reasoning + HIC + dual alignment) |

TaTS is simpler and more broadly compatible, while VoT achieves deeper semantic integration through LLM reasoning[^src-language-in-the-flow-of-time].

### vs. UniCA

| Dimension | TaTS | [[unica|UniCA]] |
|-----------|------|-------|
| Modality scope | Text only | Categorical + Image + Text |
| Core mechanism | Concatenation as variables | Covariate homogenization + attention fusion |
| TSFM integration | Any TS model | Specific TSFMs (TimesFM, Chronos) |
| Fusion strategy | Pre-input concatenation | Pre-Fusion + Post-Fusion |

### vs. Aurora

| Dimension | TaTS | [[aurora|Aurora]] |
|-----------|------|--------|
| Paradigm | Plug-in framework | Generative foundation model |
| Modalities | Text + TS | Text + Image + TS |
| Generation | Deterministic | Flow Matching (probabilistic) |
| Zero-shot | ✗ | ✓ |

### vs. Chronos

TaTS handles **text modality** through auxiliary variable concatenation, while [[chronos|Chronos]] is **numerical-only** and uses tokenization to convert time series into discrete tokens for language model processing[^src-language-in-the-flow-of-time].

### vs. Covariate-based Methods

TaTS also outperforms traditional covariate-based approaches that process text as static features rather than time-aligned variables:

| Dimension | TaTS | N-BEATS/N-HiTS | TCN | ChatTime | GPT4MTS |
|-----------|------|----------------|-----|----------|---------|
| Text handling | Auxiliary variables (time-aligned) | Static covariates | Static features | LLM chat interface | Multi-scale text features |
| Architecture change | None | Requires covariate input | Requires covariate input | Custom LLM pipeline | Custom architecture |
| Temporal alignment | ✓ (concatenation preserves position) | ✗ | ✗ | Partial | Partial |

TaTS's advantage over these methods comes from preserving the temporal position of texts — treating them as variables that evolve with the time series rather than as static contextual features[^src-language-in-the-flow-of-time].

## Related Pages

- [[source-language-in-the-flow-of-time]] — source summary
- [[chronological-textual-resonance]] — CTR phenomenon
- [[tt-wasserstein]] — TT-Wasserstein metric
- [[texts-as-auxiliary-variables]] — core design concept
- [[multimodal-time-series-forecasting]] — task concept
- [[vot]] — VoT comparison
- [[mindts]] — MindTS comparison
- [[unica]] — UniCA comparison
- [[aurora]] — Aurora comparison
- [[chronos]] — Chronos comparison
- [[heterogeneous-covariates]] — covariate classification concept

[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]