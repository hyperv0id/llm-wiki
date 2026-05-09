---
title: "Mamba ↔ Linear Attention Unified Framework"
type: concept
tags:
  - mamba
  - linear-attention
  - unified-framework
  - architecture-analysis
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Mamba ↔ Linear Attention Unified Framework

Han et al. (NeurIPS 2024) 提出的统一框架，将 Mamba（选择性状态空间模型）和线性注意力 Transformer 纳入同一数学形式中，揭示了 Mamba 本质上是线性注意力的一个变体[^src-demystify-mamba-linear-attention-2024]。

## 数学对应关系

### 线性注意力的循环形式

线性注意力使用核函数 $\phi$ 替代 softmax，其循环形式为[^src-demystify-mamba-linear-attention-2024]：

$$S_i = S_{i-1} + K_i^\top V_i, \quad Z_i = Z_{i-1} + K_i^\top$$
$$y_i = \frac{Q_i S_i}{Q_i Z_i}$$

### Mamba 的 SSM 形式

Mamba 的选择性 SSM 为[^src-demystify-mamba-linear-attention-2024]：

$$h_i = \widetilde{A}_i \odot h_{i-1} + B_i(\Delta_i \odot x_i)$$
$$y_i = C_i h_i + D \odot x_i$$

展开为累积形式：

$$y_i = C_i \sum_{j=1}^i (\prod_{k=j+1}^i \widetilde{A}_k) B_j (\Delta_j \odot x_j) + D \odot x_i$$

### 组件对应表

| Mamba 组件 | 线性注意力对应 | 功能描述 |
|-----------|---------------|---------|
| $C_i$ | $Q_i$ | 输出/查询投影 |
| $B_j$ | $K_j^\top$ | 输入/键投影 |
| $\Delta_j \odot x_j$ | $V_j$ | 值（带输入门控） |
| $\prod_{k=j+1}^i \widetilde{A}_k$ | 隐式恒等（1） | 遗忘门（线性注意力中无） |
| $D \odot x_i$ | 无对应 | 捷径残差连接 |
| $Q_i Z_i$ 分母 | Mamba 中缺失 | 注意力归一化 |

## 六项关键差异

基于统一框架，论文识别出 Mamba 与线性注意力的六项差异，并通过消融实验量化其影响[^src-demystify-mamba-linear-attention-2024]：

| 差异 | 对准确率影响 | 对吞吐量影响 |
|------|-------------|-------------|
| 输入门 $\mathbf{\Delta}_i$ | +0.9% | 1152→106 im/s |
| 遗忘门 $\widetilde{\mathbf{A}}_i$ | +0.8% | 1152→743 im/s |
| 捷径连接 $D \odot x_i$ | +0.2% | 几乎不变 |
| 无归一化 | **-5.2%** | — |
| 单头 | 边际/负面 | — |
| 块设计 | **+1.8%** | — |

## 核心洞见

1. **遗忘门可被位置编码替代**：遗忘门提供局部偏置和输入依赖的位置信息，可通过 LePE、CPE、RoPE 等位置编码实现，从而避免循环计算[^src-demystify-mamba-linear-attention-2024]
2. **归一化至关重要**：移除归一化导致 -5.2% 的灾难性下降，说明 Mamba 通过其他机制（如遗忘门的衰减效应）间接补偿了归一化
3. **块设计贡献最大**：修改的块设计（+1.8%）是 Mamba 在视觉任务中成功的最重要因素

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[mila|MILA]] — 基于此框架的线性注意力模型
- [[forget-gate-in-sequential-models|遗忘门在序列模型中的作用]]
- [[mamba-block-design|Mamba 块设计]]
- [[linear-attention-bias|线性注意力偏置（ALiBi）]]

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
