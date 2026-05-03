---
title: "Endogenous Text Alignment (ETA)"
type: technique
tags:
  - multimodal-time-series
  - text-alignment
  - contrastive-learning
  - trend-seasonal-decomposition
created: 2026-05-03
last_updated: 2026-05-03
source_count: 2
confidence: high
status: active
---

# Endogenous Text Alignment (ETA)

**Endogenous Text Alignment (ETA)** is a representation-level alignment technique introduced in [[vot|VoT]] (ICLR 2026) that establishes deep semantic alignment between time series patterns and their textual representations using decomposed pattern extraction and decomposed contrastive learning[^src-event-driven-ts-forecasting].

## Motivation

Endogenous text — structured textual descriptions derived from time series statistics (mean, frequency, etc.) — provides useful context but largely overlaps with information already present in the time series. The challenge is to extract complementary semantic information from this text and align it with temporal patterns at a fine-grained level[^src-event-driven-ts-forecasting].

## Mechanism

### 1. Time Series and Text Encoding

- Time series $\mathbf{X} \in \mathbb{R}^{L \times N}$ is encoded to temporal representations $\mathbf{H}^{\text{ts}}$
- Endogenous text $\mathbf{T}^{\text{en}}$ (statistical descriptors converted to text) is encoded via LLM to text embeddings $\mathbf{H}^{\text{text}}$[^src-event-driven-ts-forecasting]

### 2. Decomposed Pattern Extraction

ETA uses dual-query attention to separately extract trend and seasonal information from text:

$$\mathbf{E}^{\text{tr}} = \text{Attention}(\mathbf{Q}^{\text{tr}}, \mathbf{H}^{\text{text}}, \mathbf{H}^{\text{text}}), \quad \mathbf{E}^{\text{se}} = \text{Attention}(\mathbf{Q}^{\text{se}}, \mathbf{H}^{\text{text}}, \mathbf{H}^{\text{text}})$$

Where $\mathbf{Q}^{\text{tr}}, \mathbf{Q}^{\text{se}}$ are learnable queries for trend and seasonal components[^src-event-driven-ts-forecasting].

Then, cross-attention aligns temporal representations with extracted textual components:

$$\mathbf{Z}^{*} = \text{Cross-Attention}(\text{Proj}(\mathbf{H}^{\text{ts}}), \mathbf{E}^{*}, \mathbf{E}^{*}), \quad *\in\{\text{tr},\text{se}\}$$

### 3. Decomposed Contrastive Learning

Contrastive loss is computed separately for trend and seasonal component pairs:

$$\mathcal{L}_{\text{align}} = \frac{1}{2}\sum_{*\in\{\text{tr},\text{se}\}} \left(-\log\frac{\exp(\text{sim}(\bar{\mathbf{H}}^{*}_i, \bar{\mathbf{Z}}^{*}_i))}{\sum_{j=1}^{B}\exp(\text{sim}(\bar{\mathbf{H}}^{*}_i, \bar{\mathbf{Z}}^{*}_j))} - \log\frac{\exp(\text{sim}(\bar{\mathbf{Z}}^{*}_i, \bar{\mathbf{H}}^{*}_i))}{\sum_{j=1}^{B}\exp(\text{sim}(\bar{\mathbf{Z}}^{*}_i, \bar{\mathbf{H}}^{*}_j))}\right)$$

This ensures corresponding trend and seasonal components from both modalities are aligned in a shared representation space[^src-event-driven-ts-forecasting].

## Comparison with MindTS's Alignment

| Dimension | VoT (ETA) | [[mindts|MindTS]] ([[fine-grained-time-text-semantic-alignment|Fine-grained Alignment]]) |
|-----------|-----------|------|
| Decomposition axis | Trend vs. Seasonal | Endogenous (per-patch) vs. Exogenous (shared) |
| Query mechanism | Dual learnable queries | Endogenous text as query, exogenous as key/value |
| Contrastive pairs | Trend pairs + Seasonal pairs | Time-text pairs at same patch index |
| Text source | Endogenous only (statistical) | Endogenous + Exogenous |
| Output | Aligned trend/seasonal representations | Fused text representation |

Both methods use contrastive learning for alignment, but ETA decomposes along the **trend/seasonal** axis while MindTS decomposes along the **endogenous/exogenous** axis[^src-event-driven-ts-forecasting]. ETA's decomposition is more directly tied to time series intrinsic properties.

## Comparison with TaTS's Approach

| Dimension | VoT (ETA) | [[tats|TaTS]] |
|-----------|-----------|------|
| Alignment strategy | Decomposed contrastive learning | No explicit alignment (concatenation) |
| Text processing | Extract trend/seasonal from endogenous text | Encode all text → MLP → concat as variables |
| Architecture impact | Requires alignment module | No architecture modification |
| Semantic depth | Deep (component-level) | Shallow (variable-level) |
| Compatibility | VoT-specific | Any TS model |

TaTS avoids explicit alignment entirely by treating text as auxiliary variables, relying on the time series model's own capacity to learn cross-variable relationships. ETA provides deeper semantic integration but is tied to VoT's architecture[^src-language-in-the-flow-of-time].

## Related Pages

- [[vot]] — the VoT model
- [[multi-level-alignment]] — the overarching alignment concept
- [[adaptive-frequency-fusion]] — prediction-level alignment (AFF)
- [[fine-grained-time-text-semantic-alignment]] — MindTS's alignment technique (compare)
- [[content-condenser-reconstruction]] — MindTS's downstream technique
- [[tats]] — TaTS framework (simpler alternative)
- [[texts-as-auxiliary-variables]] — TaTS's core design concept

[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]
