---
title: "Cluster-Based Gating"
type: technique
tags:
  - mixture-of-experts
  - gating-function
  - pretraining
created: 2026-07-20
last_updated: 2026-07-20
source_count: 1
confidence: medium
status: active
---

# Cluster-Based Gating

簇基门控（Cluster-Based Gating）是 [[moirai-moe|Moirai-MoE]] (ICML 2025) 提出的 MoE 门控函数，利用预训练 dense 模型的 token 嵌入聚类中心来引导专家分配，替代随机初始化的线性门控[^src-moirai-moe]。

## 动机

标准 MoE 门控使用随机初始化的线性层 $G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$（Shazeer et al., 2017; Jiang et al., 2024），从头学习路由映射。但随机初始化可能导致次优的专家分配，尤其在训练早期。受 Sparse Upcycling (Komatsuzaki et al., 2022) 和 Qwen 等从 dense checkpoint 初始化 MoE 的启发，簇基门控利用已有预训练模型的表示知识为专家分配提供更好的归纳偏置。

## 技术细节

1. **预训练 dense 模型**：先训练一个 Moirai dense 模型（单输入/输出投影层，移除频率偏置）
2. **提取簇中心**：对预训练数据（LOTSA）逐层执行推理，提取注意力输出 $\tilde{x}^l \in \mathbb{R}^{T \times D}$，以 mini-batch k-means 聚类，持续更新每层的簇中心 $C^l \in \mathbb{R}^{M \times D}$（$M$ = 专家总数）
3. **MoE 预训练时路由**：每个 token 计算到各簇中心的欧氏距离，作为 token-to-expert 亲和度：$G(\tilde{x}^l) = \text{Softmax}(\text{TopK}(\text{Euclidean}(\tilde{x}^l, C^l)))$[^src-moirai-moe]

## 效果

消融实验显示，簇基门控在所有专家数量配置下一致优于线性门控和线性门控+负载均衡两种变体。Moirai-MoE-S 使用簇基门控的聚合 MAE 约为 0.64-0.65（32 experts），明显优于线性门控的约 0.66-0.67。聚类方法对齐了数据的真实表示分布，使得专家专业化更有效[^src-moirai-moe]。

## 局限性

簇基门控需要额外的前置步骤（dense 模型预训练 + 聚类中心提取），增加了训练流程的复杂性。簇中心的"冻结"性质可能限制了在 MoE 训练过程中对数据分布变化的适应能力。

[^src-moirai-moe]: [[source-moirai-moe]]
