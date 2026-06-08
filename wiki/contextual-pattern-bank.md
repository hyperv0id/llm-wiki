---
title: "Contextual Pattern Bank"
type: technique
tags:
  - continual-learning
  - spatio-temporal
  - parameter-expansion
  - prompt-based-guidance
  - pattern-memory
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Contextual Pattern Bank

The **Contextual Pattern Bank** is a trainable, incrementally expandable parameter memory introduced by [[stbp|STBP]] (ICLR 2026) for continual spatio-temporal forecasting[^src-stbp]. It stores node-specific spatio-temporal patterns as purely parametric representations, enabling continual adaptation to evolving data without requiring historical data replay or full model retraining.

## Design

The pattern bank at period $\tau$ is a matrix $\mathbf{P}_\tau \in \mathbb{R}^{N_\tau \times d}$, composed of three trainable sub-components[^src-stbp]:

| Component | Role | Interaction |
|-----------|------|-------------|
| $\mathbf{P}_\tau^{(0)}$ | Gating base | Element-wise multiplied with backbone hidden states: $\mathbf{H}_\tau \cdot (1 + \mathbf{P}^{(0)}_\tau)$ |
| $\mathbf{P}_\tau^{(1)}$ | Scaling factor | Multiplies the gating output: $\mathbf{P}^{(1)}_\tau \cdot h_\theta(\dots)$ |
| $\mathbf{P}_\tau^{(2)}$ | Attention key | Injected as additional keys in DLGA's dual-stream linear attention |

The interaction with the backbone's hidden representation $\mathbf{H}_\tau$ follows the **Prompt-Based Guidance** mechanism[^src-stbp]:

$$\mathbf{H}'_\tau = \mathbf{P}^{(1)}_\tau \cdot h_\theta\left(\mathbf{H}_\tau \cdot (1 + \mathbf{P}^{(0)}_\tau)\right)$$

This gating enables **adaptive modeling of node heterogeneity**—different nodes receive different amounts of modulation from the pattern bank[^src-stbp].

## Incremental Expansion

When new nodes are added (e.g., new sensors in an expanding traffic network), the pattern bank expands via concatenation[^src-stbp]:

$$\mathbf{P}'_\tau = \mathbf{P}_{\tau-1} \parallel \Delta\mathbf{P}_\tau$$

where $\Delta\mathbf{P}_\tau \in \mathbb{R}^{(N_\tau - N_{\tau-1}) \times d}$ are freshly initialized parameters for the new nodes. Only the expanded portion is fine-tuned; the frozen backbone ensures zero forgetting of previously learned patterns[^src-stbp].

## Key Properties

### Autonomous Clustering

Without any explicit clustering constraints or supervision, the pattern bank autonomously organizes nodes into meaningful clusters[^src-stbp]:

- Nodes within the same cluster exhibit similar periodic and trend patterns in their traffic data (**relevance**)
- Different clusters capture distinct behavioral regimes (**heterogeneity**)
- New nodes from later periods are correctly grouped into existing clusters, demonstrating **generalization** of learned patterns

This emergent behavior is driven purely by the prediction task—the pattern bank learns to distinguish nodes because doing so improves forecasting accuracy[^src-stbp].

### Privacy Preservation

Since the pattern bank encodes high-level abstractions (not raw historical data), it supports knowledge retention without revisiting prior data, offering advantages in privacy protection and storage efficiency[^src-stbp].

### Advantages over Alternatives

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| Historical replay (TrafficStream) | Replay past data | Privacy risk, storage cost |
| Expand-and-compress (EAC) | Dynamic prompt pool | Compression may lose historical information |
| Pattern matching (PECPM) | Representative pattern bank | Requires conflict detection heuristics |
| **Contextual Pattern Bank (STBP)** | Pure parametric incremental expansion | Only parameter expansion, no compression — more completely preserves historical knowledge[^src-stbp] |

## Comparison with EAC's Prompt Pool

Unlike [[eac|EAC]]'s expand-and-compress prompt pool, STBP's pattern bank[^src-stbp]:

1. Adopts **pure parametric incremental expansion** without compression, avoiding historical information loss during compression steps
2. Employs **structured multi-component design** ($\mathbf{P}_\tau^{(0)}, \mathbf{P}_\tau^{(1)}, \mathbf{P}_\tau^{(2)}$) that jointly models node relevance and heterogeneity through gating and attention mechanisms
3. Provides richer interaction with the backbone (gating + attention keys vs. EAC's simpler feature addition)

## Related Pages

- [[stbp]] — The STBP framework that introduces this technique
- [[continual-spatio-temporal-forecasting]] — The CSTF paradigm
- [[eac]] — EAC, the expand-and-compress prompt pool approach
- [[pecpm]] — PECPM, pattern-matching-based pattern bank
- [[trafficstream]] — TrafficStream, replay-based CSTF

[^src-stbp]: [[source-stbp]]
