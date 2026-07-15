---
title: "Multi-Granularity Sea Ice Forecasting"
type: concept
tags:
  - sea-ice-forecasting
  - multi-granularity
  - spatio-temporal
  - climate
  - time-scale
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Multi-Granularity Sea Ice Forecasting

**多粒度海冰预测**是指同时利用多个时间粒度（日、周、月尺度）的海冰密集度（SIC）数据进行联合建模，以提升各粒度预测性能的范式，由 [[sifusion|SIFusion]]（NeurIPS 2025）首次系统提出[^src-sifusion]。

## 动机

此前 SIC 预测方法（IceNet、SICNet、IceFormer）在固定单一时间粒度上建模，仅利用 **intra-granularity** 信息，忽略了不同粒度之间的 **inter-granularity** 相关性[^src-sifusion]：

- **累积效应**：短期日尺度波动长期累积可改变季节趋势
- **一致性约束**：不同粒度的 SIC 预测应自然一致——月平均不能与构成它的日值矛盾
- **双向促进**：长期趋势提供先验约束，辅助短期预测校准；精细粒度提供更准确的初始条件，辅助季节预测

## SIFusion 的多粒度方案

SIFusion 联合三个时间粒度[^src-sifusion]：

| 粒度 | 输入长度 | 预测长度 | 覆盖尺度 |
|------|---------|---------|---------|
| 日尺度 | 7 天 | 7 天 | sub-seasonal |
| 周平均 | 8 周 | 8 周 | sub-seasonal |
| 月平均 | 6 月 | 6 月 | seasonal |

三粒度覆盖从 sub-seasonal 到 seasonal 尺度，恰好填补 [[subseasonal-to-seasonal-forecasting|S2S 预测]] 的"可预测性荒漠"[^src-sifusion]。

## 关键机制：Granularity Variates

同一粒度的全部时间步 SIC 经空间编码后拼接为一个 **granularity variate**，三个粒度形成三个 variate。Encoder-only Transformer 的 attention 在 variate 维度上运作，显式捕获跨粒度（inter-granularity）相关性，而 FFN 独立处理各 variate 的 intra-granularity 序列[^src-sifusion]。

## 验证

消融实验（Table 2）表明，多粒度联合训练相比单一粒度分别训练，在所有三个粒度的全部指标上均有显著提升——验证了 SIC 数据内在跨粒度相关性的存在和可利用性[^src-sifusion]。

## 与通用多尺度建模的区别

传统多尺度建模（如 [[multi-scale-attention|多尺度注意力]]、[[multi-scale-linear-prediction|多尺度线性预测]]）关注同一序列内不同分辨率特征的提取；多粒度海冰预测的核心创新在于**将不同时间粒度的 SIC 视为独立的 variate 跨粒度建模**，而非在同一序列内做多分辨率分解[^src-sifusion]。

## 相关页面

- [[sifusion]] — SIFusion 模型
- [[granularity-variates]] — 粒度 variate 技术
- [[independent-spatial-tokenization]] — 独立空间 tokenization
- [[sea-ice-concentration-forecasting]] — 海冰密集度预测
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测
- [[multi-scale-attention]] — 多尺度注意力

[^src-sifusion]: [[source-sifusion]]
