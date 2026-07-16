---
title: "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts"
type: source-summary
tags:
  - time-series
  - foundation-model
  - mixture-of-experts
  - pretraining
  - forecasting
created: 2026-07-20
last_updated: 2026-07-25
source_count: 1
confidence: high
status: active
---

# Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts

**Authors**: Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao Liu, Junnan Li, Silvio Savarese, Caiming Xiong, Doyen Sahoo (Salesforce AI Research, NUS, HKUST-GZ). ICML 2025, PMLR 267.

## 核心问题

时间序列基础模型（TSFM）统一预训练的核心挑战在于数据的高度异质性。Moirai (Woo et al., 2024) 通过为不同频率设计独立输入/输出投影层实现频率级专业化，TimesFM 则使用频率嵌入映射。但本文指出频率不是可靠的数据分组指标——不同频率的时间序列可呈现相似模式（反之亦然），且单个时间序列内部也呈现非平稳分布变化。频率级专业化在粒度上忽略了这种多样性。

## 核心方法：Moirai-MoE

Moirai-MoE 移除人为定义的频率级投影层，改为在 Transformer 内部引入稀疏混合专家（Sparse MoE）实现**数据驱动的 token 级专业化**。每个 FFN 层被替换为包含 M=32 个专家的 MoE 层，每个 token 仅激活 K=2 个专家。核心创新包括：

1. **Token 簇门控函数**：先从预训练 dense Moirai 模型提取注意力输出，以 mini-batch k-means 聚类得到每层的簇中心点 C_l，然后以 token 到各簇中心的欧氏距离作为专家分配亲和度分数，而非通常的随机初始化线性门控。这使得专家分配更贴合数据真实分布。

2. **Next-token prediction 预训练目标**：以负对数似然 NLL 预测下一 patch 的混合分布参数，替代 Moirai 的掩码填充目标，提升预训练并行效率。

3. **统一输入/输出投影**：去除按频率分组的多个投影层，所有频率数据共享单一投影层，让 MoE Transformer 学习处理模式多样性。

## 实验

在 Monash 29 个数据集的 in-distribution 评估中，Moirai-MoE-S (11M 激活参数) 超越 dense Moirai-S 17%，优于 Moirai-B 和 Moirai-L 7-8%。在 10 个零样本数据集上，Moirai-MoE-B 取得最佳 CRPS 和 MASE，以 28× 更少激活参数超越 Moirai-L。相比 Chronos-L，以 65× 更少激活参数取得更优结果。消融表明主要增益来自 MoE token 级专业化，预处理目标变化贡献较小。

## 模型分析核心发现

1. Moirai-MoE 实现**频率不变隐表示**——浅层 expert 分配在不同频率间各异，深层（layer 6）趋于一致。
2. 模型执行**渐进式去噪**——浅层多用多个 expert 处理短程变异性（周期/季节/突变），深层集中在可共享的通用趋势。
3. 此行为与 LLM 中浅层集中、深层分散的模式相反，源于时间序列 token 的动态和噪声特性。

## 局限

部分 expert 在推理时很少被选中，剪枝低利用率 expert 以提升推理效率留待未来工作。Moirai-MoE-L 因计算资源需求未实现。
