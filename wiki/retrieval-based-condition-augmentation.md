---
title: "Retrieval-Based Condition Augmentation"
type: technique
tags:
  - retrieval-augmented-generation
  - condition-augmentation
  - cross-city-transfer
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Retrieval-Based Condition Augmentation (RCA)

**Retrieval-Based Condition Augmentation (RCA)** is a lightweight plug-in module proposed in [[craft|CRAFT]] (NeurIPS 2025) for enriching diffusion model conditions in cross-city traffic flow generation. It addresses the **insufficient condition** problem: static geographic features alone cannot capture the temporal dynamics (stochastic properties like mean, peak, variance) needed for realistic flow generation in unseen cities[^src-craft].

## Motivation

In cross-city scenarios, the target city has no historical flow data. Regions with similar geography may share similar **periodicity and trends** (e.g., morning/evening peaks), but their **absolute magnitudes** (mean flow volume, peak height, variance) can differ dramatically[^src-craft]. These dynamic properties cannot be inferred from static features alone.

RCA bridges this gap by **retrieving relevant historical data from source cities** and using it to augment the diffusion model's conditioning input[^src-craft].

## Components

### Time Embedding ($t_{emb}$)

A temporal encoding composed of three periodic components[^src-craft]:
- $t_{month} \in [1, 12]$ — month in year
- $t_{day} \in [1, 7]$ — day in week
- $t_{hour} \in [1, 24]$ — hour in day

Each component is separately embedded and concatenated: $t_{emb} = (t_{month} \| t_{day} \| t_{hour})$. This captures the periodic patterns independent of geography[^src-craft].

### Flow Retrieval ($x_{i,t}$)

For each target region $r_i$ at time $t$, RCA retrieves relevant historical flow segments from source cities using two filtering criteria[^src-craft]:

1. **Temporal filter**: Match by $t_{month}$, $t_{day}$, $t_{hour}$ — ensures periodicity alignment
2. **Spatial filter**: Match by geographic representation similarity between $h_i$ (target) and $h_j$ (source) — ensures functional similarity

The top-K matched flow sequences $\bar{X}_{i,t} = \{\bar{X}_{1,t}, ..., \bar{X}_{K,t}\}$ are aggregated via self-attention[^src-craft]:

$$x_{i,t} = \text{Attn}\left( \frac{1}{K} \sum_{k=1}^{K} \bar{X}_{k,t} \right)$$

Averaging before attention mitigates the impact of retrieval noise[^src-craft].

### Condition Fusion

The final diffusion condition is[^src-craft]:

$$c_{i,t} = \text{MLP}(h_i \| x_{i,t} \| t_{emb})$$

Three information sources are combined: geographic context ($h_i$), retrieved dynamics ($x_{i,t}$), and temporal signal ($t_{emb}$).

## Key Findings

- RCA consistently improves performance, validating that retrieved dynamics from source cities meaningfully supplement the conditioning for target city generation[^src-craft]
- **Temporal embedding ($t_{emb}$) contributes most** within RCA — periodic temporal patterns are the most transferable signal across cities[^src-craft]
- **Retrieved features ($x_{i,t}$) provide additional gain** — dynamic patterns supplement what static features cannot capture[^src-craft]
- RCA is a **lightweight plug-in**: no backbone architecture changes, operates purely at the condition level[^src-craft]

## Ablation

| Ablation | Effect |
|----------|--------|
| w/o RCA (remove $x_{i,t}$) | Notable performance drop — confirms value of retrieved dynamics |
| w/o Temporal Embed (remove $t_{emb}$) | Largest drop within RCA — periodic patterns are critical[^src-craft] |
| No ablation (full RCA) | Best performance across all metrics[^src-craft] |

## Relationship to Other Techniques

- **vs. standard RAG** (NLP): RCA retrieves from source cities' historical flow rather than a text corpus; filtering is dual-dimension (temporal + spatial similarity)[^src-craft]
- **vs. [[rast|RAST]]** (AAAI 2026): RAST uses RAG for traffic **prediction** with FAISS-indexed spatio-temporal memory banks; RCA is designed for cross-city flow **generation** with geographic representation-based retrieval[^src-craft]
- **vs. [[retrieval-guidance|MiDDiR retrieval guidance]]**: MiDDiR uses retrieval at diffusion inference time for score tilting; RCA uses retrieval at condition construction time for input augmentation[^src-craft]

## See Also

- [[craft]] — CRAFT model overview
- [[geographic-feature-alignment]] — GFA, complement to RCA
- [[cross-city-traffic-flow-generation]] — problem domain
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF paradigm

[^src-craft]: [[source-craft]]
