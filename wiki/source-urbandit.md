---
title: "UrbanDiT: Diffusion Transformers as Open-World Spatiotemporal Foundation Models"
type: source-summary
tags:
  - spatiotemporal
  - diffusion-model
  - transformer
  - foundation-model
  - traffic-forecasting
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: medium
status: active
---

**UrbanDiT** 是一种基于 Diffusion Transformer（DiT）架构的开放世界时空基础模型，由清华大学 FIB Lab 提出，发表于 NeurIPS 2025[^src-urbandit].

代码仓库：https://github.com/tsinghua-fib-lab/UrbanDiT [^src-urbandit].

## 核心思路

UrbanDiT 的核心在于统一处理多种城市时空数据，通过 DiT backbone + prompt learning，在开放世界场景中实现多种时空预测任务的统一建模[^src-urbandit].

模型架构包含四个关键组件：

1. **数据统一化** — 将不同类型的城市时空数据（交通流量、人群流动、出租车需求、共享单车使用、蜂窝网络流量等）统一到相同的表示空间[^src-urbandit]
2. **扩散管道** — 采用扩散模型的加噪-去噪框架进行生成式预测[^src-urbandit]
3. **任务指定掩码** — 通过不同的掩码策略指定不同的任务类型[^src-urbandit]
4. **统一提示学习** — 结合数据驱动的提示和任务特定的提示来增强去噪过程[^src-urbandit]

## 支持的任务

UrbanDiT 明确支持四种时空预测任务[^src-urbandit]：

| 任务 | 描述 |
|------|------|
| 双向时空预测 | 同时预测过去和未来的时空状态 |
| 时间插补 | 填补时间维度上的缺失值 |
| 空间外推 | 从已知区域推断未知区域的时空模式 |
| 时空插补 | 同时填补时间和空间维度的缺失数据 |

## 零样本能力

UrbanDiT 的核心优势在于其强大的零样本泛化能力——在未见过的城市或场景上，其性能超越多数有训练数据的基线模型[^src-urbandit].

适用领域涵盖：交通流量、人群流动、出租车需求、共享单车使用、蜂窝网络流量等城市计算场景[^src-urbandit].

## 技术实现

模型实现采用 PyTorch，依赖 torch == 2.0.0[^src-urbandit]。训练流程示例：


## 评估数据集

模型在多个城市、多个领域的数据集上进行了评估，具体数据访问方式参见仓库中的 data 文档[^src-urbandit].

## 局限性

论文未在开源仓库中明确报告的具体数据规模、计算资源需求和精确的性能数值。

[^src-urbandit]: 用户提供的论文摘要 + [GitHub 仓库 README](https://github.com/tsinghua-fib-lab/UrbanDiT) (2026-05-12)
