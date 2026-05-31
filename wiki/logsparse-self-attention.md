---
title: "LogSparse Self-Attention"
type: technique
tags:
  - attention
  - sparse-attention
  - efficient-transformer
  - time-series
  - long-sequence
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# LogSparse Self-Attention

**LogSparse self-attention** is an efficient attention mechanism introduced in [[logtrans|LogTrans]] (Li et al., NeurIPS 2019) that reduces Transformer memory complexity from $O(L^2)$ to $O(L (\log L)^2)$ by restricting each cell's attention to exponentially spaced historical positions[^src-logtrans].

## Mechanism

### Core Idea

In standard self-attention, cell $l$ attends to all prior cells $\{j : j \le l\}$, producing a dense $L \times L$ attention matrix with $O(L^2)$ memory. LogSparse replaces this with an exponentially spaced subset[^src-logtrans]:

$$I_l^k = \{l - 2^{\lfloor \log_2 l \rfloor}, l - 2^{\lfloor \log_2 l \rfloor - 1}, l - 2^{\lfloor \log_2 l \rfloor - 2}, \ldots, l - 2^0, l\}$$

Each cell attends to at most $\lfloor \log_2 L \rfloor + 1$ cells per layer (itself plus exponentially spaced predecessors), giving $O(L \log L)$ memory per layer[^src-logtrans].

### Information Flow Guarantee

**Theorem 1**: Stacking $\lfloor \log_2 L \rfloor + 1$ layers guarantees that every cell receives information from all prior cells. Moreover, for $j < l$, the number of distinct paths from cell $j$ to cell $l$ grows at $O(\lfloor \log_2(l-j) \rfloor !)$ — super-exponential in the log-distance[^src-logtrans].

The proof constructs a path by decomposing the binary representation of $l - j = \sum b_m 2^m$: the path takes steps corresponding to each 1-bit, in order. Total path length equals the number of 1-bits, which is at most $\lfloor \log_2 (l-j) \rfloor + 1$. Reordering the steps generates factorial-many distinct paths[^src-logtrans].

This means: the model uses dramatically less memory *per layer* but stacks slightly deeper to compensate, and between distant cells the "bandwidth" of information flow actually grows richer (not poorer) due to path multiplicity.

### Extensions

Two optional refinements, both preserving $O(L(\log L)^2)$ complexity[^src-logtrans]:

- **Local attention**: Each cell densely attends to a window of $O(\log L)$ immediate left neighbors before applying LogSparse for the rest. This captures fine-grained local trends that exponential steps might skip.
- **Restart attention**: The input sequence of length $L$ is divided into sub-sequences, and LogSparse is applied independently within each sub-sequence (restarting the exponential step pattern). This reduces maximum path length and increases path count.

Local and restart attention can be combined.

## Complexity

| Mechanism | Per-layer Memory | Total Memory | Pattern |
|-----------|-----------------|--------------|---------|
| Canonical self-attention | $O(L^2)$ | $O(J \cdot L^2)$ | Dense |
| **LogSparse** | $O(L \log L)$ | $O(L (\log L)^2)$ | Fixed exponential |
| [[probsparse-self-attention|ProbSparse]] | $O(L \log L)$ | $O(L \log L)$ (with distilling) | Data-driven |

ProbSparse achieves better total complexity through distilling and data-driven query selection, but LogSparse was the first to demonstrate that structured sparsity with provable information flow guarantees could break the quadratic bottleneck for time series[^src-zhou-informer-2021].

## Implementation

The current implementation uses a mask matrix: a boolean attention mask with exponential-step True entries, applied before softmax. This is simple but does not realize the theoretical memory savings at the implementation level (the full attention matrix is still materialized then masked). Custom CUDA kernels for block-sparse operations are noted as potential optimization[^src-logtrans].

## Empirical Validation

On electricity-f (370 series × 129,120 time steps, 15-min granularity) and traffic-f (963 series × 12,435 time steps, 20-min granularity)[^src-logtrans]:

- **Same memory budget**: LogSparse with input length 768 attends to 112 cells max per cell; the full attention counterpart with length 293 attends to all 293. On traffic-f, LogSparse achieves *better* $R_{0.5}$ (0.150 vs 0.161), confirming that covering a longer history with sparse attention beats shorter full attention when long-term dependencies dominate.
- **Same input length**: Full attention (all 768 cells) performs better than LogSparse (112/768 cells) as expected on most datasets. However, on traffic-f with strong long-term dependencies, LogSparse + convolutional self-attention ($R_{0.5}=0.138$) slightly outperforms canonical full attention ($0.147$), suggesting that the combination of local context awareness and sparse long-range access is more effective than indiscriminate full attention[^src-logtrans].

## Relationship to Other Sparse Attention Methods

| Method | Pattern | Data-Dependent? | Provenance |
|--------|---------|----------------|------------|
| **LogSparse** | Fixed exponential intervals | No | [[logtrans|LogTrans]] (NeurIPS 2019) |
| [[probsparse-self-attention|ProbSparse]] | Top-$u$ queries by KL sparsity | Yes | [[informer|Informer]] (AAAI 2021) |
| Sparse Transformer | Fixed stride + local windows | No | Child et al. (2019) |
| Reformer (LSH) | Hash-bucketed queries/keys | Yes | Kitaev et al. (2020) |

LogSparse's key distinction: it is the only method with a **constructive proof** that information flows between any pair given sufficient depth, combined with super-exponential path multiplicity for distant pairs[^src-logtrans].

[^src-logtrans]: [[source-logtrans]]
[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
