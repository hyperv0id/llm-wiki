---
title: "Stochastic Shared Embedding (SSE)"
type: technique
tags:
  - regularization
  - embedding
  - stochastic
  - graph-neural-network
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Stochastic Shared Embedding (SSE)

SSE 是一种嵌入层正则化技术，由 Wu et al. (NeurIPS 2019) 提出，最初用于推荐系统[^src-ragc-efficient-traffic-forecasting]。RAGC 将其引入交通预测领域，用于正则化节点嵌入。

## 机制

对于节点嵌入矩阵 $E \in \mathbb{R}^{N \times d}$，每个节点 $i$ 的嵌入 $e_i$：

1. 采样 Bernoulli 掩码：$m_i \sim \text{Bernoulli}(p)$
2. 随机选择替换索引：$r(i) \sim \text{Uniform}\{1, 2, \ldots, N\}$
3. 生成扰动嵌入：$\tilde{e}_i = m_i e_{r(i)} + (1 - m_i) e_i$

期望值：$\mathbb{E}[\tilde{e}_i] = p \bar{e} + (1-p) e_i$，其中 $\bar{e} = \frac{1}{N}\sum_{j=1}^{N} e_j$ 是全局平均嵌入[^src-ragc-efficient-traffic-forecasting]。

- $p = 0$：无扰动，$\tilde{e}_i = e_i$
- $p > 0$：每个嵌入保留自身信息加上全局平均的分数

## 正则化原理

通过注入全局平均噪声，SSE 迫使每个节点减少对自身嵌入的依赖，促进嵌入学习更鲁棒的表示。在推荐系统中，随机替换嵌入引入有益的语义混合，帮助模型探索潜在关系[^src-ragc-efficient-traffic-forecasting]。

## 交通预测中的挑战

交通流量预测是**时空预测任务，对连续性和数值精度要求极高**。SSE 随机替换节点嵌入会在输入信号中注入时空噪声，通过残差连接和多层层传播，扰动会扭曲学习的时间序列并跨层累积，最终破坏训练稳定性[^src-ragc-efficient-traffic-forecasting]。

具体分析：在 L 层残差网络中，扰动嵌入 $\tilde{e}_i$ 通过残差路径直接传播到输出层（见公式 $H^{(L)} = \tilde{e}_i + \sum_{l=1}^{L} F^{(l)}(H^{(l)})$），噪声持续影响所有层。

## RAGC 的解决方案

RAGC 将 SSE 与 [[residual-difference-mechanism|残差差分机制]]结合，使图卷积的权重矩阵 $W_g$ 可以自适应地抑制 SSE 引入的全局平均噪声。当权重和 $\tilde{W}_g = \sum_{z=0}^{Z} W^{(z)}_g$ 趋近单位矩阵时，噪声项被有效消除[^src-ragc-efficient-traffic-forecasting]。

## 超参数选择

替换概率 $p$ 的选择[^src-ragc-efficient-traffic-forecasting]：
- $p = 0.1$：在 SD、GBA、CA 数据集上最优
- $p = 0.2$：在 GLA 数据集上最优
- $p$ 过高会导致过度正则化，阻碍模型拟合训练数据

## 与其他正则化方法对比

| 方法 | 正则化目标 | 交通预测适用性 |
|------|-----------|---------------|
| SSE | 嵌入层 | 需配合噪声抑制机制（如 RDM） |
| Dropout | 激活层 | 破坏空间信息完整性 |
| Laplacian 正则化 | 嵌入距离 | 依赖预定义图，缺乏数据驱动灵活性 |

## 相关页面

- [[ragc]] — 将 SSE 引入交通预测的模型
- [[residual-difference-mechanism|RDM]] — 抑制 SSE 噪声的残差差分机制
- [[node-embedding-regularization]] — 节点嵌入正则化概念

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
