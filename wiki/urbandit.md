---
title: "UrbanDiT"
type: entity
tags:
  - spatiotemporal
  - foundation-model
  - diffusion-transformer
  - traffic-forecasting
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: medium
status: active
---

**UrbanDiT**（Urban Diffusion Transformer）是清华大学 FIB Lab 提出的开放世界时空基础模型，基于 Diffusion Transformer（DiT）架构，统一处理多种城市时空预测任务[^src-urbandit].

## 核心特点

- **统一数据表示** — 将交通流量、人群流动、出租车需求、共享单车使用、蜂窝网络流量等多种数据类型统一到相同表示空间[^src-github]
- **多任务支持** — 同时支持双向时空预测、时间插补、空间外推、时空插补四项任务[^src-urbandit]
- **零样本泛化** — 在未见过的城市或场景上，性能超越多数有训练数据的基线模型[^src-urbandit]

## 架构组件

模型包含四个关键组件[^src-github]：
1. **数据统一化模块** — 类型到统一表示的转换
2. **扩散管道** — 基于扩散模型的加噪-去噪预测框架
3. **任务指定掩码** — 通过掩码策略指定不同任务类型
4. **统一提示学习** — 结合数据驱动和任务特定的提示增强去噪过程

## 相关资源

- 论文：[[source-urbandit]] — NeurIPS 2025
- 代码仓库：https://github.com/tsinghua-fib-lab/UrbanDiT

[^src-urbandit]: [[source-urbandit]]
