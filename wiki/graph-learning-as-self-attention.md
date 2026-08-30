---
title: "Graph Learning as Self-Attention"
type: concept
tags:
  - graph-neural-network
  - self-attention
  - transformer
  - interpretability
created: 2026-07-16
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Graph Learning as Self-Attention

**Graph Learning as Self-Attention** 是一种将图学习模块数学等价于 Transformer 中自注意力机制的范式，由 Qi et al. (ICML 2026) 在混合图算法展开的工作中明确提出[^src-lightweight-mixed-graph-unrolling]。

## 核心对应

经典自注意力的注意力权重计算[^src-lightweight-mixed-graph-unrolling]：

$$a_{i,j} = \frac{\exp(e(i,j))}{\sum_l \exp(e(i,l))}, \quad e(i,j) = (\mathbf{Q}\mathbf{x}_j)^\top (\mathbf{K}\mathbf{x}_i)$$

图学习模块中的边权重计算[^src-lightweight-mixed-graph-unrolling]：

$$w_{i,j}^u = \frac{\exp(-d^u(i,j))}{\sqrt{\sum_{l \in \mathcal{N}_i} \exp(-d^u(i,l)) \cdot \sum_{k \in \mathcal{N}_j} \exp(-d^u(k,j))}}$$

其中马氏距离 $d^u(i,j) = (\mathbf{f}_i^u - \mathbf{f}_j^u)^\top \mathbf{M} (\mathbf{f}_i^u - \mathbf{f}_j^u)$。

## 等价性

将马氏距离的负值 $-$d^u(i,j)$ 解释为负能量 $-$e(i,j)$，则图边权重 $w_{i,j}^u$ 即为注意力权重 $a_{i,j}$。这意味着[^src-lightweight-mixed-graph-unrolling]：

- 特征提取函数 $\mathbf{F}^u(\cdot)$ 取代了 Q/K 投影
- PSD 度量矩阵 $\mathbf{M}$ 取代了 $\mathbf{Q}^\top\mathbf{K}$（但参数量远小于两个 $E\times E$ 矩阵）
- 不需要值矩阵 $\mathbf{V}$：输出通过低通滤波器（展开的 ADMM 层）计算，而非加权求和

## 参数效率

这是关键优势：在标准 Transformer 中，Q、K、V 矩阵贡献了主要参数量；在图学习自注意力中，参数缩减为[^src-lightweight-mixed-graph-unrolling]：

- 紧凑的特征提取函数 $\mathbf{F}^u(\cdot), \mathbf{F}^d(\cdot)$（如浅层 GraphSAGE）
- 较小的 PSD 度量矩阵 $\mathbf{M}, \mathbf{P} \in \mathbb{R}^{K\times K}$（K=6）
- 无 V 矩阵

这使整体模型参数量仅 38K，为 [[pdformer|PDFormer]]（1,404K）的约 2.7%（论文声称 7.2%，但 38/1404≈2.7%；论文存在数值不一致）。

## 多头扩展

类似于标准 Transformer 的多头注意力，该框架通过学习多组度量矩阵 $\mathbf{M}^{(h)}$ 并行构造多个混合图（H=4），各图经独立 ADMM block 处理后通过可学习融合层合并输出[^src-lightweight-mixed-graph-unrolling]。这种 [[algorithm-unrolling|算法展开]] 范式使每层具有明确的优化解释。

## 差异化特征

| 特征 | 标准自注意力 | 图学习自注意力 |
|------|----------|-------------|
| Query/Key | $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{E\times E}$ | $\mathbf{F}(\cdot) + \mathbf{M} \in \mathbb{R}^{K\times K}$ |
| Value | $\mathbf{V} \in \mathbb{R}^{E\times E}$ | 无需 V（用 ADMM 低通滤波替代） |
| 注意力模式 | 全局 | 限制于 k-NN/时间窗口邻域 |
| 解释性 | 黑盒 | 边权重 $\leftrightarrow$ 物理距离 |
| 方向性 | 无向 | 可自然支持有向图（DGL） |

## 参考文献

[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
