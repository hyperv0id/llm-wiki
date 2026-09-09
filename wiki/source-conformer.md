---
title: "ConFormer: Conditional Transformer for Accident-Informed Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - transformer
  - accident-aware
  - kdd-2026
created: 2026-04-28
last_updated: 2026-09-06
source_count: 1
confidence: medium
status: active
---

# ConFormer: Conditional Transformer for Accident-Informed Traffic Forecasting

**Authors**: Hongjun Wang, Jiawei Yong, Jiawei Wang, Shintaro Fukushima, Renhe Jiang (The University of Tokyo, Toyota Motor Corporation)

**Venue**: KDD 2026, Jeju Island, Republic of Korea
**arXiv**: [2512.09398](https://arxiv.org/abs/2512.09398) · **Code**: <https://github.com/Dreamzz5/ConFormer>

## Core Contribution

ConFormer addresses a critical gap in traffic forecasting: existing models excel at capturing recurring patterns but fail when traffic accidents create non-stationary perturbations with directional shockwaves through transportation networks[^src-conformer].

## Key Innovations

1. **Accident-Aware Graph Propagation** — Models how disruptions spread asymmetrically through traffic networks using graph convolution with K-hop Laplacian operations[^src-conformer].

2. **Guided Layer Normalization (GLN)** — Replaces static LayerNorm parameters with dynamic affine transformations (γ, β) conditioned on traffic conditions, enabling adaptive feature transformations across normal and accident scenarios[^src-conformer].

3. **Conditional Self-Attention** — Extends vanilla self-attention to incorporate contextual condition representations, with residual connections modulated by learned factor α[^src-conformer].

## Datasets

Two enriched large-scale benchmark datasets:
- **Tokyo**: 1,843 highway segments, Oct-Dec 2021, 10-min intervals, accident + regulation data from JARTIC
- **California**: Bay Area (2,352 sensors) + San Diego (716 sensors), 2019, 15-min intervals, accident data from US Accidents database[^src-conformer]

## Performance

ConFormer consistently outperforms state-of-the-art models including STAEFormer:
- Tokyo: 1.7% MAE improvement, 21.5% MAPE improvement in accident scenarios
- San Diego: 4.7% MAE improvement, 5.0% MAPE improvement
- Bay Area: 1.8% MAE improvement
- Up to **10.7% improvement** in accident scenarios specifically[^src-conformer]

## Theoretical Insight

GLN enables adaptive feature transformations through condition-dependent affine parameters. The scaling factor γ controls sensitivity to abrupt changes (higher γ → rapid accident adaptation), while the shifting factor β emphasizes node-specific features during disruptions[^src-conformer].

## Graph Propagation 细节（2026-09-06 arXiv 全文核对）

§4.1 Eq.(2)：$X^c = \texttt{GraphPropagation}(X^o, \mathcal{A}) = [X^o \,\|\, \mathcal{L}X^o \,\|\, \dots \,\|\, \mathcal{L}^K X^o]$——对融合嵌入施加图拉普拉斯 $\mathcal{L} \in [0,1]^{N \times N}$ 的幂并沿特征维拼接，$\{\mathcal{L}^k\}_{k=0}^K$ 构成多项式滤波器基；$X^c$ 经 MLP 生成 GLN 的 $(\beta, \gamma, \alpha)$。

- 相对 GCN 原型 $\sigma(\sum_{k=0}^{K}\theta_k \mathcal{L}^k X)W$：论文文字明说省略特征混合 $W$，Eq.(2) 中逐跳权重 $\theta_k$ 与激活 $\sigma$ 也未出现。论文未给 $\mathcal{L}$ 显式归一化公式，$[0,1]$ 取值与 renormalized 传播算子一致（推断）。
- 超参研究（东京数据集，图 7）：传播阶数 1–2 已有效捕捉空间依赖，更高阶收益有限；拼接使特征维膨胀 $(K{+}1)$ 倍。
- 「不对称传播（upstream 影响大于 downstream）」出自引言对事故冲击波的动机描述（Wu et al. 2019、Zhang et al. 2020），非 Eq.(2) 算子本身的性质。

## Limitations

- Requires accident/regulation data integration
- Single attention layer may limit very long-range dependencies
- Graph propagation order (K) adds computational overhead

[^src-conformer]: [[source-conformer]]