---
title: "Fine-grained Time-text Semantic Alignment"
type: technique
tags:
  - multimodal-time-series
  - text-alignment
  - contrastive-learning
  - cross-view-fusion
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Fine-grained Time-text Semantic Alignment

**Fine-grained Time-text Semantic Alignment** is a technique introduced in [[mindts|MindTS]] (ICLR 2026) for achieving semantically consistent alignment between time series and text modalities at the patch level[^src-multimodal-ts-anomaly-detection].

## Motivation

Time series and text reside in fundamentally different semantic spaces: time series are continuous numerical signals with temporal dependencies, while text is discrete and semantic. Previous approaches either:

- Generate **endogenous text** from time series using LLMs (e.g., Time-LLM, LLMMixer), which ensures alignment but offers limited semantic richness
- Incorporate **exogenous text** (news, reports) as external context, which provides rich background but is difficult to align with specific time segments[^src-multimodal-ts-anomaly-detection]

MindTS addresses this by combining both text views and explicitly aligning them with time series.

## Three Components

### 1. Endogenous Text Generation

For each time patch $P_i \in \mathbb{R}^{p \times D}$, a unified prompt template generates statistical descriptions:

- Mean, extrema (max/min), overall trend (upward/downward)
- Applied per-patch rather than globally, matching the dynamic property of time series

The endogenous text $O = \{o_1, o_2, ..., o_N\}$ is encoded by an LLM-based text encoder to produce $H^O_{\text{text}} \in \mathbb{R}^{N \times d_{\text{model}}}$[^src-multimodal-ts-anomaly-detection].

### 2. Cross-view Text Fusion

Exogenous text $C$ (shared across all patches) is encoded to $H^C_{\text{text}} \in \mathbb{R}^{1 \times d_{\text{model}}}$. Cross-view attention selectively extracts complementary information:

$$Z_{\text{text}} = \text{LayerNorm}(\hat{Z}_{\text{text}} + \text{FeedForward}(\hat{Z}_{\text{text}}))$$

$$\hat{Z}_{\text{text}} = \text{LayerNorm}(H^O_{\text{text}} + \text{CrossAttn}(H^O_{\text{text}}, H^C_{\text{text}}, H^C_{\text{text}}))$$

Endogenous text serves as **query** (time-specific), exogenous text as **key/value** (background knowledge). This ensures the fused representation is both contextually rich and temporally precise[^src-multimodal-ts-anomaly-detection].

### 3. Multimodal Alignment via Contrastive Learning

A contrastive loss explicitly aligns time representations $H_{\text{time}}$ and fused text representations $Z_{\text{text}}$:

$$L_{\text{MA}} = -\frac{1}{2N} \left[ \sum_{j=1}^{N} \log \frac{\exp(k(h^j_{\text{time}}, z^j_{\text{text}})/\tau)}{\sum_{g=1}^{N} \exp(k(h^j_{\text{time}}, z^g_{\text{text}})/\tau)} + \sum_{g=1}^{N} \log \frac{\exp(k(h^g_{\text{time}}, z^g_{\text{text}})/\tau)}{\sum_{j=1}^{N} \exp(k(h^j_{\text{time}}, z^g_{\text{text}})/\tau)} \right]$$

Positive pairs are time-text pairs from the same patch index $(j=g)$. The loss pulls positive pairs closer and pushes negative pairs apart in the shared embedding space[^src-multimodal-ts-anomaly-detection].

## Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| Per-patch endogenous text | Captures fine-grained temporal dynamics, avoids global semantic drift |
| Cross-view attention (endo→exo) | Endogenous text guides selection of relevant background knowledge |
| Contrastive alignment | Explicitly bridges the modality gap, unlike additive/concatenative fusion |
| Shared exogenous text | Ensures background context is not lost due to limited patch scope |

## Ablation Evidence

Removing the time-text semantic alignment module causes a notable performance drop, confirming that effective modality alignment is essential for reliable anomaly detection[^src-multimodal-ts-anomaly-detection].

## Related Pages

- [[mindts]] — the MindTS model
- [[content-condenser-reconstruction]] — downstream redundancy filtering
- [[multimodal-time-series-anomaly-detection]] — task concept
- [[cross-view-text-fusion]] — cross-view attention mechanism
- [[contrastive-learning]] — contrastive learning paradigm
- [[endogenous-text-alignment]] — VoT's ETA technique (compare: trend/seasonal decomposition vs. endo/exo decomposition)
- [[vot]] — VoT model (same lab, forecasting task)
- [[tats]] — TaTS framework (no explicit alignment, treats text as auxiliary variables)
- [[chronological-textual-resonance]] — CTR phenomenon (observed periodicity alignment, not engineered)

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]