---
title: "Continuous Prompt Parameter Pool"
type: technique
tags:
  - prompt-learning
  - continual-learning
  - spatio-temporal
  - parameter-efficient-tuning
  - low-rank-approximation
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# Continuous Prompt Parameter Pool

The **continuous prompt parameter pool** is a node-level, dynamically expandable set of learnable parameters introduced by [[eac|EAC]] (ICLR 2025) for continual spatio-temporal graph forecasting[^src-eac]. It serves as the sole trainable component during continual adaptation, while the backbone STGNN remains frozen — a design that completely eliminates catastrophic forgetting without historical data replay.

## Mechanism

### Structure

The prompt pool P is a matrix (or set of matrices) where each row corresponds to a node-specific prompt vector. At the initial period τ=1, each node i receives a learnable parameter vector p_i ∈ R^d, forming P^(1) ∈ R^(n×d). These prompts are fused with input features via **element-wise addition**[^src-eac]:

$$X'_τ = X_τ + P$$

The fused representation X' is then fed into the frozen STGNN backbone for training and prediction.

### Expansion

When new nodes appear in period τ, a fresh prompt matrix A^(τ) is created for only the new nodes and appended to the pool. The backbone is frozen throughout; only the expanded pool is optimized on current-period data. This enables the model to accommodate growing graphs without modifying the backbone architecture[^src-eac].

### Compression

To mitigate parameter inflation from continuous expansion, EAC decomposes the prompt pool into low-rank form: P^(τ) ≈ A^(τ)B, where A^(τ) ∈ R^(n×k), B ∈ R^(k×d), and k ≪ d (default k=6). B is shared across all periods; only A grows with node count, reducing tunable parameters to ~59% of the full-rank equivalent at k=6[^src-eac].

## Design Rationale

### Why Node-Level?

Node-level prompts exploit the **heterogeneity** inherent in spatio-temporal sensor data: different sensors exhibit distinct behavioral patterns (e.g., highway vs. arterial road traffic). STGNNs naturally capture spatial correlations via message passing but struggle to express node-specific deviations. The prompt pool explicitly parameterizes these deviations, expanding the feature space's dispersion (measured by Average Node Deviation)[^src-eac].

### Why Low-Rank?

Spectral analysis of trained prompt pools reveals strong low-rank structure: >75% cumulative singular value concentration in the top few components across all periods. This redundancy arises because many nodes share similar underlying patterns. The low-rank decomposition A^(τ)B exploits this, reducing parameters while preserving representational capacity[^src-eac].

### Why Separate from Backbone?

Storing the prompt pool in memory (separate from the backbone) enables:
- **Zero forgetting**: Backbone never changes; old knowledge is perfectly preserved
- **Efficient adaptation**: Only ~59% of parameters tuned vs full-model fine-tuning
- **Flexible deployment**: Pool can be loaded/unloaded per period without touching backbone weights[^src-eac]

## Comparison to Related Techniques

| Technique | Scope | Backbone | Expansion | Compression |
|-----------|-------|----------|-----------|-------------|
| EAC Prompt Pool | Node-level | Frozen | Add new rows | Low-rank AB^T |
| [[contextual-pattern-bank|STBP Pattern Bank]] | Node-level | Frozen | Add new rows | None (pure expansion) |
| [[unified-prompt-learning|UrbanDiT Memory Pool]] | Domain-level | Trainable | Fixed size | Not applicable |
| LoRA (for CSTF) | Layer-level | Trainable | Fixed rank | Inherent in design |

The key distinction: EAC's pool is the **only** trainable component, operating at input level rather than being injected into intermediate layers. [[stbp|STBP]] later adopted the same frozen-backbone + expandable-pool paradigm but dropped compression, achieving stronger performance through a richer backbone.

## Limitations

- Parameter count still grows with nodes, albeit at reduced rate (O(nk) vs O(nd)). The paper acknowledges this and suggests sparsification/pruning as future directions[^src-eac].
- Low-rank compression introduces approximation error; k trades off performance vs. efficiency (Figure 7 in the paper)[^src-eac].
- Currently demonstrated only in graph-expansion scenarios; node contraction handling is claimed but not empirically validated[^src-eac].

[^src-eac]: [[source-eac]]
