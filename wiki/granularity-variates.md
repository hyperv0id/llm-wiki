---
title: "Granularity Variates"
type: technique
tags:
  - variate-modeling
  - multi-granularity
  - transformer
  - sequential-modeling
  - sea-ice-forecasting
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Granularity Variates

**Granularity variates** 是 [[sifusion|SIFusion]] 多粒度融合的核心技术：将同一时间粒度的全部 SIC 观测经空间编码后拼接为一个 variate token 序列，多个粒度形成多个独立的 variate，通过 encoder-only Transformer 的 attention 在 variate 维度上捕获跨粒度（inter-granularity）相关性[^src-sifusion]。

## 灵感来源

Granularity variates 的灵感来自 [[itransformer|iTransformer]]（ICLR 2024）的变量维度 attention：iTransformer 将每个时间序列变量作为一个 token，让 attention 作用在变量维度以捕获多变量相关性。SIFusion 将这一思路迁移到时间粒度维度——每个粒度成为一个 variate，attention 捕获的是跨粒度而非跨变量的相关性[^src-sifusion]。

## 机制对比

SIFusion 比较了三种时序建模 backbone[^src-sifusion]：

| 机制 | 描述 | 问题 |
|------|------|------|
| **Vanilla Transformer** | Attention 作用在 temporal token，FFN 作用在 variate | 长 lookback 窗口性能退化、多变量 token 产生不连贯 attention map |
| **MLP-Mixer** | token-mixing（时间维）+ channel-mixing（variate 维），交替 Transpose | 无法显式建模序列信息 |
| **SIFusion（granularity variate）** | Attention 作用在 variate token（inter-granularity），FFN 独立处理各 variate（intra-granularity） | 匹配海冰跨粒度累积效应的物理直觉 |

消融实验（Table 3）表明，SIFusion 的 granularity variate attention 在全部粒度和指标上显著优于 Vanilla Transformer 和 MLP-Mixer 替代方案[^src-sifusion]。

## 构造流程

1. 各粒度 SIC 独立通过共享 Swin Transformer V2 空间编码器 → 每个时间步生成 1D spatial token（经由 linear projection）
2. 同一粒度内 spatial token 按时间顺序拼接 → 形成该粒度的 variate token 序列
3. 周/月粒度序列通过 linear transformation 对齐到日粒度序列长度
4. 三个 variate 送入 encoder-only Transformer：每层先做 inter-variate attention，再做 per-variate FFN[^src-sifusion]

## 与相关技术

| 技术 | 关系 |
|------|------|
| [[itransformer|iTransformer]] | Granularity variates 灵感来源——将时间粒度视为类似变量的建模单元 |
| [[multivariate-correlation-attention|多变量相关性注意力]] | 同一机制在变量维度的应用 |
| [[multi-scale-attention|多尺度注意力]] | 同属多粒度建模，但 multi-scale attention 在单序列内做多窗口，granularity variates 跨不同粒度独立 variate 联合 |

[^src-sifusion]: [[source-sifusion]]
