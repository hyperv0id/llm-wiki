---
title: "Token-Level Specialization"
type: concept
tags:
  - time-series
  - mixture-of-experts
  - pretraining
  - foundation-model
created: 2026-07-20
last_updated: 2026-07-25
source_count: 2
confidence: high
status: active
---

# Token-Level Specialization

Token 级专业化（Token-Level Specialization）是 [[moirai-moe|Moirai-MoE]] (ICML 2025) 提出的时间序列基础模型设计范式，是对现有频率级专业化（如 Moirai 的多频率投影层、TimesFM 的频率嵌入映射）的根本改进[^src-moirai-moe]。

## 动机

时间序列预训练数据的核心挑战是异质性：不同频率的时间序列可呈现相似模式（反之亦然），且单个序列内部也在短时间内呈现分布变化。频率作为分组指标是不可靠的先验——它未必反映数据的真实语义结构。

现有方案沿两个方向组织专业化：

| 策略 | 代表方法 | 粒度 |
|------|---------|------|
| 数据集级 | UniTime、TEMPO（语言 prompt 标识数据源） | 粗 |
| 频率级 | Moirai（多频率投影层）、TimesFM（频率嵌入） | 中 |
| 无专业化 | Chronos、Lag-LLaMA、Timer | 无 |

Token 级专业化是更细粒度的替代：不预设任何分组，让模型通过稀疏 MoE 自动学习哪些 token 应共享参数空间、哪些应分配给不同专家。

## 工作机制

在 Moirai-MoE 中，每个 Transformer 层的 FFN 被替换为 M 个专家的 MoE 层，每个 token 仅激活 K 个专家（K≪M）。门控函数基于 token 与预计算簇中心的距离决定路由。这使得语义相似的 token 自然分配到相同专家，不依赖人为元数据标签。

## 优势

1. **频率不变性**：Moirai-MoE 深层 expert 分配在不同频率间趋于一致，证明模型提取了超越频率的高级表示[^src-moirai-moe]
2. **渐进式去噪**：浅层用多 expert 处理局部变异性，深层集中在通用趋势——token 级粒度允许这种逐层抽象[^src-moirai-moe]
3. **性能提升**：仅 11M 激活参数即超越 310M 参数的 Moirai-L，17% Monash 聚合 MAE 提升[^src-moirai-moe]

## 与 Time-MoE 的对比

[[time-moe|Time-MoE]] (ICLR 2025) 比 Moirai-MoE 更早将 Sparse MoE 引入时序基础模型，但采用不同的专业化范式：使用标准线性门控 + Top-K 路由 + 辅助负载均衡损失[^src-time-moe]。Time-MoE 对专家的路由是通过与输入 token 无关的线性参数 $W_i^l$ 计算的——这意味着专业化是由输入数据的统计分布隐式驱动的，而非通过显式的簇分配[^src-time-moe]。

这种差异反映了两种专业化哲学：（1）Time-MoE 的隐式数据驱动方式——让辅助均衡 loss 在训练中自然地塑造专家分工；（2）Moirai-MoE 的显式结构驱动方式——通过预训练 dense 模型的嵌入聚类来显式引导专家分配[^src-moirai-moe]。两者都实现了 token 级的细粒度专业化，但路由机制和初始化策略不同。

频率级专业化的根本局限在于：频率是元数据标签而非语义标签。Moirai-MoE 通过实验证明，不同频率但相似模式的数据集（如 NN5 Daily 和 Traffic Hourly）在频率级投影下嵌入分离，而在 token 级 MoE 下融合，且最终 expert 分配分布相似——这正是 token 级专业化的核心优势[^src-moirai-moe]。

[^src-moirai-moe]: [[source-moirai-moe]]
[^src-time-moe]: [[source-time-moe]]
