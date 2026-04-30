---
title: "TimesFM"
type: entity
tags:
  - time-series
  - foundation-model
  - decoder-only
  - iclr2024
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# TimesFM

## 概述

**TimesFM** (Time-Series Forecasting) 是 Google 提出的首个用于时间序列预测的解码器-only基础模型，发表于 ICLR 2024[^src-timesfm]。

## 核心贡献

1. **大规模预训练**：在超过 10 亿时间点的大规模数据上预训练
2. **Patch 级别的处理**：将时间序列划分为重叠的 patch，作为 token
3. **Decoder-only 架构**：类似 LLM 的自回归预测
4. **强零样本能力**：在未见过的数据集上表现出色

## 架构

```
Input Patches → Linear Embedding → Transformer Decoder → Output Projection
                                                        ↑
                                            [Autoregressive]
```

### 关键设计

- **Patch Size**：128 个时间点
- **Context Length**：2048 个时间点（512 个 patch）
- **预���长度**：可变
- **模型规模**：200M 参数

## 与 UniCA 的结合

TimesFM 可以通过 UniCA 框架适配到协变量感知预测任务：

- **Pre-Fusion**：融入历史协变量
- **Post-Fusion**：融入未来已知协变量
- **性能提升**：+12.7% MAPE (单模态)，+6.5% MAE (多模态图像)[^src-unca]

## 相关页面

- [[unified-covariate-adaptation]] — UniCA 框架
- [[timesnet]] — 另一个时间序列基础模型
- [[chronos]] — Chronos 时间序列模型

---

## 引用

[^src-timesfm]: [[source-timesfm]]
[^src-unca]: [[source-unca]]