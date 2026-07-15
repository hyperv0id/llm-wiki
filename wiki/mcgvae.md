---
title: "McgVAE"
type: entity
tags:
  - traffic-forecasting
  - lane-level
  - multi-granularity
created: 2026-07-16
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# McgVAE

**McgVAE**（Multi-Channel Graph-structured Variational Autoencoder）是首个道路-车道联合建模的细粒度交通预测模型，由 Li et al. 提出（CIKM 2024），通过多通道集成架构在道路和车道两个粒度上分别捕捉时空信息[^src-minitraffic]。

## 架构

McgVAE 采用集成架构，通过变分自编码器将道路级信息整合为车道级预测的全局视角[^src-minitraffic]。其多通道结构分别处理不同粒度的时空信息：道路级编码、车道级编码、以及道路-车道交叉信息融合。

## 在 MiniTraffic 中的角色

在 [[minitraffic|MiniTraffic]]（ICML 2026）的实验中，McgVAE 是细粒度多任务预测的主要基线，也是 29 个基线中唯一已有的多任务细粒度模型[^src-minitraffic]。MiniTraffic 在车道级预测上全面超越 McgVAE：PeMS-Lane MAE 降低 7%–24%，HuaNan-Lane MAE 降低 24%–39%[^src-minitraffic]。

McgVAE 的核心局限：需要同时输入道路级和对应车道级数据、无预训练机制（每个新场景需从头训练）、参数量较大（~544K vs MiniTraffic 的 119K）[^src-minitraffic]。

## 相关

- [[fine-grained-traffic-prediction]] — 细粒度交通预测问题域
- [[minitraffic]] — MiniTraffic（ICML 2026），以 McgVAE 为主要基线的轻量预训练替代方案

[^src-minitraffic]: [[source-minitraffic]]
