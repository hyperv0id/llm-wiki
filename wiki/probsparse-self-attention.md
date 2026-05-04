---
title: "ProbSparse Self-Attention"
type: technique
tags:
  - attention
  - efficient-transformer
  - long-sequence
  - sparse-attention
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# ProbSparse Self-Attention

**ProbSparse Self-Attention** is an efficient attention mechanism proposed in the [[informer|Informer]] model (Zhou et al., AAAI 2021 Best Paper) that reduces the time and memory complexity of canonical dot-product self-attention from $O(L^2)$ to **$O(L \log L)$**[^src-zhou-informer-2021].

## Motivation

In canonical self-attention, every query attends to every key. However, empirical analysis reveals that the attention score distribution follows a **long-tail pattern**: only a handful of query-key pairs contribute meaningfully to the output, while the vast majority produce near-uniform (uninformative) attention weights. Computing full attention over all pairs is therefore wasteful — most computations yield negligible contributions[^src-zhou-informer-2021].

## Mechanism

### Step 1: Sparsity Measurement

For each query $q_i$, Informer computes a **sparsity score** $M(q_i, K)$ based on the KL divergence between the query's attention distribution and the uniform distribution:

$$M(q_i, K) = \max_j\left\{\frac{q_i k_j^\top}{\sqrt{d}}\right\} - \frac{1}{L_K}\sum_{j=1}^{L_K}\frac{q_i k_j^\top}{\sqrt{d}}$$

- Higher $M$ → the query's attention is concentrated (sparse, informative)
- Lower $M$ → the query's attention is near-uniform (dense, can be approximated by a mean)

### Step 2: Query Selection

Only the **Top-$u$ queries** with the highest sparsity scores are selected for full attention computation, where $u = c \cdot \ln L_Q$ (controlled by the sampling factor $c$). This is $O(L \ln L)$ in both time and memory.

### Step 3: Efficient Implementation

The empirical max-mean measurement requires computing all dot-products, which is $O(L^2)$. To avoid this, Informer samples $U = L_K \ln L_Q$ dot-product pairs (reducing complexity to $O(L \ln L)$) and uses the top-$u$ from these as approximate estimates. In practice, this approximation incurs negligible performance degradation.

### Step 4: Sparse Attention Output

For the selected queries, full attention is computed as normal. For the remaining (discarded) queries, the **mean of the value vectors** replaces the attention output — since their attention distributions are near-uniform, the mean is a good approximation.

## Complexity Comparison

| Mechanism | Time | Memory |
|-----------|------|--------|
| Canonical Self-Attention | $O(L^2)$ | $O(L^2)$ |
| LogSparse Attention | $O(L(\log L)^2)$ | $O(L \log L)$ |
| Reformer (LSH) | $O(L \log L)$ | $O(L \log L)$ |
| **ProbSparse Attention** | **$O(L \log L)$** | **$O(L \log L)$** |

ProbSparse achieves the same asymptotic complexity as LSH-based methods, but through a fundamentally different (and arguably simpler) data-driven mechanism.

## Distinction from Other Sparse Attention Methods

ProbSparse differs from earlier sparse attention approaches in a crucial way:
- **Sparse Transformer**: Uses fixed stride patterns — sparsity pattern is architecturally hard-coded, not data-dependent.
- **LogSparse**: Uses exponentially growing intervals — while attention scope grows with distance, the specific pattern is still fixed.
- **Reformer (LSH)**: Uses locality-sensitive hashing to bucket similar queries/keys — effective but requires hash collision luck.
- **ProbSparse**: The sparsity pattern is **fully data-driven** — which queries are "important" depends on the actual dot-product scores computed from the input, adapting to each sequence's specific long-range dependency structure.

## Usage in Informer

ProbSparse attention replaces canonical self-attention in both the encoder and decoder of [[informer|Informer]]. In the encoder, it is followed by a self-attention distilling operation. In the decoder, it is combined with causal masking to enable the generative (one-forward-pass) prediction paradigm without autoregressive leakage.

[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]