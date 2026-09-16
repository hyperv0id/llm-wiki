---
title: "Graph Frequency Decomposition"
type: concept
tags:
  - graph-neural-network
  - spectral-methods
  - frequency-domain
  - graph-signal-processing
created: 2026-06-08
last_updated: 2026-09-16
source_count: 2
confidence: medium
status: active
---

# Graph Frequency Decomposition

Graph frequency decomposition refers to the explicit separation of graph signals into low-frequency (smooth, global) and high-frequency (sharp, local) components, enabling neural networks to model both coarse structural semantics and fine-grained variations simultaneously[^src-hifinet].

## Motivation

In graph signal processing, the graph Laplacian eigenbasis defines a notion of "frequency": eigenvectors with small eigenvalues correspond to smooth (low-frequency) signals, while those with large eigenvalues correspond to oscillatory (high-frequency) signals[^src-hifinet]. Most GNNs—GCN, GAT, GraphSAGE—inherently act as **low-pass filters**, smoothing node features across neighborhoods and discarding high-frequency information[^src-hifinet]. While this aids denoising, it causes [[over-smoothing-in-gnns|over-smoothing]] and loss of local discriminative patterns.

## HiFiNet's Decomposition-Updating-Reconstruction Paradigm

[[hifinet|HiFiNet]] (AAAI 2026) introduces a structured three-stage framework for graph frequency decomposition[^src-hifinet]:

### 1. Decomposition
Through hierarchical graph construction (segment → locality → region), the model performs top-down low-pass filtering via GAT-based unpooling. The resulting low-frequency features $H_S^l$ capture smooth global trends. High-frequency features are extracted by subtraction[^src-hifinet]:

$$H_S^h = H_S - H_S^l$$

### 2. Updating
Both $H_S^l$ and $H_S^h$ are independently refined through a [[topology-aware-graph-transformer|Topology-Aware Graph Transformer (TGT)]] that blends global self-attention with local graph topology[^src-hifinet].

### 3. Reconstruction
The updated components are fused with a learnable coefficient[^src-hifinet]:

$$\\hat{H}_S = \\beta \\cdot \\tilde{H}_S^l + (1-\\beta) \\cdot \\tilde{H}_S^h$$

## Theoretical Foundation

HiFiNet's Theorem 1 proves that the hierarchical projection $A_{XY}$ (segment→locality or locality→region assignment) acts as a **spectral low-pass filter**: it approximately preserves low-frequency energy while attenuating high-frequency components[^src-hifinet]. This provides theoretical justification for why the hierarchical unpooling output can be used as the low-frequency component, and subtraction yields the high-frequency component.

## Comparison to Other Frequency Methods

| Approach | Domain | Mechanism |
|----------|--------|-----------|
| ChebNet/GCN | Spectral | Polynomial filter on Laplacian eigenbasis (fixed, not learnable per-frequency) |
| [[fedformer|FEDformer]] | Time series | Fourier/Wavelet transform → attention in frequency domain |
| [[freqflow-ts|FrèqFlow]] | Time series | Conditional flow matching in spectral domain |
| **HiFiNet** | GNN on road graphs | Hierarchical assignment → spatial low-pass → subtraction → explicit LF/HF modeling |
| **[[adafre|AdaFre]]** | Spatio-temporal forecasting on sensor graphs | Laplacian eigenvectors cut into $P$ equal-count groups by eigenvalue, each aligned one-to-one with a temporal frequency band |

Unlike methods that apply frequency transforms (FFT, DCT) to time series, HiFiNet performs frequency decomposition **structurally** through hierarchical graph coarsening—the graph topology itself determines which signals are filtered as "low" vs "high" frequency[^src-hifinet].

## AdaFre 的时间-空间频率对齐

[[adafre|AdaFre]]（AAAI 2026）走的是另一条路线：它不改变图结构，而是对**已有的**拉普拉斯特征向量按特征值从小到大等量切分成 $P$ 组，让第 $p$ 组只服务时间轴的第 $p$ 个频带；时间侧同样用 DFT 把有效频率范围均匀切成 $P$ 带并逆变换回时域[^src-adafre]。因此该页讨论的"图频率分解"在 AdaFre 里是**基底的分组**而非信号的滤波：滤波发生在时间轴，空间侧只提供与该频率相匹配的先验[^src-adafre]。

两者的对照可以概括为：HiFiNet 通过层次粗化让拓扑本身决定哪些信号算低频，高频由减法得到，分组是二分的、与时间维度无关；AdaFre 的分组数与时间频带数绑定、按特征值等量切分，并为每个频带配一条独立的预测路径[^src-hifinet][^src-adafre]。AdaFre 论文未与 HiFiNet 一类结构性分解做对照实验[^src-adafre]。

## Applications

Beyond road networks, the hierarchical frequency-decomposition paradigm could apply to[^src-hifinet]:
- Social networks (communities at different scales)
- Molecular graphs (functional groups → interactions)
- Point clouds (local geometry → global shape)

[^src-hifinet]: [[source-hifinet]]
[^src-adafre]: [[source-adafre]]
