---
title: "SIFusion"
type: entity
tags:
  - sea-ice-forecasting
  - multi-granularity
  - spatio-temporal
  - transformer
  - climate
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# SIFusion

**SIFusion**（Sea Ice Fusion）是一个统一多时间粒度的北极海冰密集度（SIC）预测 Transformer 框架，由复旦大学、上海人工智能实验室和香港中文大学在 NeurIPS 2025 提出[^src-sifusion]。

## 架构

SIFusion 由三个核心组件构成[^src-sifusion]：

1. **共享空间编码器（Shared Spatial Encoder）**：Swin Transformer V2 骨干，用 2×2 patch partition 将各时间粒度（日、周平均、月平均）SIC 独立编码为 shared embedding space 中的 spatial token，生成 1D compact spatial representation。空间特征跳跃连接保留编码过程中的空间信息。

2. **多粒度融合（Multi-Granularity Fusion）**：将各粒度的 spatial token 沿自身时间维依次拼接形成 granularity variates，经线性投影对齐维度后送入 encoder-only Transformer。Attention 作用在 variate token 上显式捕获 inter-granularity 相关性，FFN 独立处理各 variate 的 intra-granularity 序列特征。Sequential feature skip connection 通过 cross-attention 补偿深层编码的信息损失。

3. **共享空间解码器（Shared Spatial Decoder）**：对称的 Swin Transformer 骨干，用 patch expanding 替代 patch merging 恢复特征图分辨率，输出多粒度未来 SIC。

## 关键设计选择

- **为什么要独立空间 tokenization？** 此前主流方法（IceNet、SICNet）用 U-Net + channel-wise fusion 隐式建模时空关系，channel expansion/contraction 干扰序列特征，且多变量 channel 混合进一步恶化时空序列建模。独立空间 tokenization 解耦空间与时间，使后续显式序列建模成为可能[^src-sifusion]。
- **为什么用 granularity variates 而非 temporal tokens？** Vanilla Transformer 在 temporal token 上做 attention 面临长 lookback 窗口性能退化，且不同物理量（多变量）混在同一 temporal token 中产生不连贯 attention map。Granularity variates 将同一粒度的全部时间步作为一个 variate，更匹配海冰的跨粒度累积效应[^src-sifusion]。

## 对比

| 特性 | U-Net 类方法 | SIFusion |
|------|-------------|----------|
| 时间粒度 | 单一固定 | 三粒度联合（日/周/月） |
| 空间编码 | channel-wise fusion | 独立 spatial tokenization |
| 时序建模 | 隐式（channel 操作） | 显式（granularity variate attention） |
| 跨粒度信息 | 无 | inter-granularity attention |

仅用 SIC 数据（无大气/海洋辅助变量），在所有粒度和六个指标上超越 IceNet、SICNet、MT-IceNet、IceFormer 等 baseline[^src-sifusion]。

## 相关页面

- [[source-sifusion]] — 论文源摘要
- [[multi-granularity-sea-ice-forecasting]] — 多粒度海冰预测概念
- [[granularity-variates]] — 粒度 variate 建模技术
- [[independent-spatial-tokenization]] — 独立空间 tokenization 技术
- [[sea-ice-concentration-forecasting]] — 海冰密集度预测领域
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测
- [[arctic-amplification]] — 北极放大效应，海冰预测气候背景

[^src-sifusion]: [[source-sifusion]]
