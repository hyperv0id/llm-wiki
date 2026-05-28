---
title: "Unified Prompt Learning"
type: technique
tags:
  - prompt-learning
  - spatiotemporal
  - memory-pool
  - diffusion-transformer
  - foundation-model
created: 2026-05-28
last_updated: 2026-05-28
source_count: 1
confidence: medium
status: active
---

**Unified Prompt Learning**（统一提示学习）是 [[urbandit|UrbanDiT]] 提出的核心创新机制，通过 learnable 的 memory pool 和任务感知的掩码 prompt，使单个模型在多数据类型和多任务场景下自适应地生成引导信号，增强去噪过程[^src-urbandit]。

## 设计动机

城市时空数据来自不同城市、不同领域（交通流量、人群流动、出租车需求等），其数据分布和时空模式差异巨大[^src-urbandit]。训练统一模型的核心挑战在于：如何让模型感知这些差异并自适应调整行为。提示学习（prompt learning）作为一种灵活的中间层机制，通过在输入端注入引导信号来解决这一问题[^src-urbandit]。

## 机制

### Data-Driven Prompt（数据驱动提示）

UrbanDiT 维护三个 learnable 的 key-value **memory pool**，分别编码时域、频域、空域的通用模式[^src-urbandit]：

```
Memory Pool 结构（每池）：

  Key   Value
 ┌────┐ ┌────┐
 │k₁  │→│v₁  │
 │k₂  │→│v₂  │
 │... │→│... │
 │k₅₁₂│→│v₅₁₂│  ← 512 个 learnable embeddings
 └────┘ └────┘

Prompt 检索（以时域为例）：
  αₜ = softmax(Xₜ, Kₜ)     ← cosine similarity 匹配
  Pₜ = Σ αₜ · Vₜ            ← 加权聚合
```

三个 memory pool 的配置[^src-urbandit]：

| Pool | 模式提取方式 | 作用 |
|------|------------|------|
| 时间域 $(K_t, V_t)$ | Temporal attention（每个空间位置独立） | 捕获时序依赖 |
| 频域 $(K_f, V_f)$ | FFT → 4 种阈值策略（无阈值/均值/分位数/Top-k） | 捕获周期性模式 |
| 空间域 $(K_s, V_s)$ | Spatial attention（每个时间 patch 独立） | 捕获空间关联 |

最终拼接：$X = \text{Concat}(P_t, P_f, X)$，将各维度模式注入输入序列[^src-urbandit]。

### Task-Specific Prompt（任务特定提示）

从任务掩码 $M$ 生成任务感知 prompt[^src-urbandit]：

$$P_m = \text{Attention}(\text{Flatten}(M))$$

不同的掩码策略（forward mask → 未来时间步、backward mask → 过去时间步、interpolation mask → 中间时间点、extrapolation mask → 未知空间区域、imputation mask → 随机位置）生成不同的 $P_m$，使模型自然地识别当前任务类型[^src-urbandit]。

最终输入格式：$X = \text{Concat}(P_t, P_f, P_m, X)$[^src-urbandit]。

### 消融结果

移除各类 prompt 对性能影响[^src-urbandit]：
- **移除频域 prompt（w/o F）**：性能下降最严重
- **移除时域 prompt（w/o T）**：性能显著下降
- **移除空间域 prompt（w/o S）**：性能明显下降
- **移除任务 prompt（w/o M）**：性能下降
- **移除所有 prompt（w/o P）**：性能最差

表明频域信息对城市时空建模最为关键，四类 prompt 互补[^src-urbandit]。

## 与其他 Prompt 方法的对比

| 方法 | Prompt 来源 | 适用场景 |
|------|-----------|---------|
| **Unified Prompt Learning** | Learnable memory pools（时域/频域/空域） + 任务掩码 | 多任务、多数据源的统一时空建模 |
| UniST 的 prompt | 时空知识引导的 learnable prompts | 单任务预测 |
| PromptST 的 prompt | 预训练 + prompt tuning | 多属性预测 |

## 相关页面

- [[urbandit]] — UrbanDiT 模型主体
- [[rectified-flow]] — 训练的加速框架
- [[spatio-temporal-foundation-model]] — 时空基础模型概念

[^src-urbandit]: [[source-urbandit]]