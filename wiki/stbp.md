---
title: "STBP"
type: entity
tags:
  - continual-learning
  - spatio-temporal
  - traffic-forecasting
  - pattern-bank
  - linear-attention
  - iclr-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# STBP

**STBP** (General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank) is a continual spatio-temporal forecasting framework proposed by Aoyu Liu and Yaying Zhang (Tongji University), accepted as a poster at ICLR 2026[^src-stbp]. It addresses the core challenge of **continual spatio-temporal forecasting (CSTF)**: how to continuously learn from streaming, evolving spatio-temporal graph data without catastrophic forgetting or full retraining.

## Architecture

STBP consists of two tightly coupled components[^src-stbp]:

### 1. General Spatio-Temporal Backbone

The backbone is designed to be node-count-independent and adjacency-matrix-free, making it adaptable to arbitrary spatio-temporal data structures. It contains:

- **FreNet (Frequency-Domain Network)**: Two FreNets (one at entry, one at exit of the backbone) transform input data to the frequency domain via FFT, apply a learnable frequency-domain embedding to emphasize stable low-frequency components (periodicity, trends), and convert back via IFFT. This suppresses high-frequency noise and provides robustness to distributional drift across incremental periods[^src-stbp].
- **DLGA (Dual-Stream Linear Graph Attention)**: A linear attention mechanism using random feature mapping ($\phi(\cdot)$) to reduce spatial complexity from $O(N^2)$ to $O(N)$. Uniquely, DLGA has a dual-stream structure: one stream models representation-based node correlations (standard QKV attention), the other models prompt-based correlations using the contextual pattern bank as additional keys. This enables the model to assess relationships between evolving input patterns and stored knowledge[^src-stbp].
- **Feedforward Layer**: MLP for enhanced nonlinear expressivity.

### 2. Contextual Pattern Bank

The pattern bank $\mathbf{P}_\tau \in \mathbb{R}^{N_\tau \times d}$ is a set of purely trainable parameters that encode node-specific spatio-temporal patterns. It consists of three components[^src-stbp]:

- $\mathbf{P}_\tau^{(0)}$: Gating component that modulates the backbone's hidden representation through element-wise multiplication and addition: $\mathbf{H}'_\tau = \mathbf{P}^{(1)}_\tau \cdot h_\theta(\mathbf{H}_\tau \cdot (1 + \mathbf{P}^{(0)}_\tau))$
- $\mathbf{P}_\tau^{(1)}$: Scaling component for the gating mechanism
- $\mathbf{P}_\tau^{(2)}$: Key embedding injected into the DLGA attention module as an additional key stream

These components enable the model to simultaneously capture **relevance** (shared behavioral patterns among similar nodes) and **heterogeneity** (differences due to function, geography, policy, events). t-SNE analysis confirms that the pattern bank autonomously forms meaningful clusters without explicit supervision[^src-stbp].

## Continual Learning Workflow

1. **Initial period ($\tau = 1$)**: Jointly train backbone and pattern bank on initial data
2. **Subsequent periods ($\tau > 1$)**: 
   - Freeze the backbone (preserving all previously learned general spatio-temporal knowledge)
   - Expand the pattern bank: $\mathbf{P}'_\tau = \mathbf{P}_{\tau-1} \| \Delta\mathbf{P}_\tau$ where $\Delta\mathbf{P}_\tau$ are new parameters for newly added nodes
   - Fine-tune only the expanded pattern bank on current-period data
3. The expanded pattern bank serves as a "prompt" guiding the frozen backbone to adapt to new distributions[^src-stbp]

This decoupling achieves parameter-efficient adaptation without historical data replay, offering privacy and storage benefits[^src-stbp].

## Performance

On three real-world streaming spatio-temporal datasets (PEMS-Stream, CA-Stream, AIR-Stream), STBP achieves SOTA, reducing average MAE by 21.44%, 21.93%, and 2.35% respectively over the best baseline EAC[^src-stbp]. It also excels in few-shot scenarios (10% training data) and maintains efficiency comparable to lightweight CSTF methods despite its richer backbone[^src-stbp].

## Relationship to Foundation Models

STBP explicitly positions itself as a step toward **spatio-temporal foundation models**. The paper's conclusion states that extending STBP's fixed-backbone + expandable-bank paradigm to cross-domain continual learning is "a crucial step towards developing a foundational spatio-temporal model"[^src-stbp]. Its approach—decoupling a stable general backbone from scalable domain-specific parameters—parallels the design philosophy of [[factost|FactoST]], [[urbanpg|UrbanPG]], and the broader [[spatio-temporal-foundation-model|ST foundation model]] paradigm.

## Related Pages

- [[contextual-pattern-bank]] — The core innovation of STBP
- [[continual-spatio-temporal-forecasting]] — The CSTF paradigm
- [[eac]] — EAC (Chen & Liang, 2025), the strongest CSTF baseline
- [[traffic-forecasting]] — Domain overview
- [[spatio-temporal-foundation-model]] — The broader foundation model context
- [[pecpm]] — PECPM, pattern-matching-based CSTF predecessor
- [[trafficstream]] — TrafficStream, first CSTF framework

[^src-stbp]: [[source-stbp]]
