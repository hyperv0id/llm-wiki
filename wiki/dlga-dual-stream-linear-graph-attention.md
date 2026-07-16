---
title: "DLGA (Dual-Stream Linear Graph Attention)"
type: technique
tags:
  - linear-attention
  - graph-attention
  - spatio-temporal
  - continual-learning
created: 2026-07-22
last_updated: 2026-07-22
source_count: 1
confidence: medium
status: active
---

# DLGA (Dual-Stream Linear Graph Attention)

**DLGA** (Dual-Stream Linear Graph Attention) is a spatial attention module introduced by [[stbp|STBP]] (ICLR 2026) for continual spatio-temporal forecasting[^src-stbp]. It combines two innovations: linear attention via random feature mapping (reducing complexity from $O(N^2)$ to $O(N)$) and a dual-stream structure that integrates prompt-based knowledge from the [[contextual-pattern-bank|contextual pattern bank]].

## Design

### Linear Attention

DLGA replaces standard softmax attention with a linear approximation using random feature mapping $\phi(\cdot)$[^src-stbp]:

$$\text{Attention}(Q, K, V, \mathbf{P}^{(2)}_\tau) \approx \phi(Q)\left(\phi(K)^\top V + \phi(\mathbf{P}^{(2)}_\tau)^\top V\right)$$

The key insight: by reordering operations, the quadratic $O(N^2)$ complexity of computing $QK^\top$ is avoided. Instead, $\phi(K)^\top V$ is computed first (an $O(N)$ operation), followed by multiplication with $\phi(Q)$[^src-stbp].

### Dual-Stream Structure

DLGA has two parallel attention streams[^src-stbp]:

1. **Representation-based stream**: Standard QKV attention on the input hidden states $\mathbf{H}^s_\tau$, modeling correlations between node representations
2. **Prompt-based stream**: Uses the contextual pattern bank component $\mathbf{P}^{(2)}_\tau$ as additional keys, enabling the model to assess relationships between evolving input patterns and stored knowledge

The combined output is:

$$\mathbf{H}^{s'}_\tau = \text{Softmax}(QK^\top + Q(\mathbf{P}^{(2)}_\tau)^\top)V$$

In the linear approximation, this becomes two separate aggregation terms[^src-stbp]:
- **Term 1** (representation-based aggregation): $\phi(Q)\phi(K)^\top V$
- **Term 2** (prompt-based aggregation): $\phi(Q)\phi(\mathbf{P}^{(2)}_\tau)^\top V$

## Key Properties

- **No explicit adjacency**: DLGA models dynamic spatial correlations implicitly through attention computations, without constructing a predefined adjacency matrix[^src-stbp]
- **Scalability**: Linear complexity enables handling of large, growing graphs (e.g., CA-Stream with 1,698 nodes)[^src-stbp]
- **Knowledge-aware**: The dual-stream design allows the pattern bank to inject stored knowledge directly into spatial attention, going beyond simple feature addition[^src-stbp]
- **Ablation significance**: Removing DLGA causes significant performance degradation, validating its role in capturing dynamic spatial correlations[^src-stbp]

## Related Pages

- [[stbp]] — The STBP framework
- [[contextual-pattern-bank]] — The pattern bank that provides prompt keys to DLGA
- [[frenet-frequency-domain-network]] — FreNet, the temporal counterpart in STBP's backbone
- [[linear-attention]] — General linear attention mechanism

[^src-stbp]: [[source-stbp]]
