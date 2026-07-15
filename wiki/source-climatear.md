---
title: "ClimateAR: Multi-Scale Autoregressive Generative Modeling for Climate Forecasting"
type: source-summary
tags:
  - climate-forecasting
  - autoregressive-model
  - generative-model
  - probabilistic-forecasting
  - vector-quantization
  - multi-scale
  - 2026
  - icml
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# ClimateAR: Multi-Scale Autoregressive Generative Modeling for Climate Forecasting

Yu, Chen, Wu, Cui, Jiang, Shang, Wu, Sun & Chen (Zhejiang University, Alibaba DAMO Academy, IAP CAS) 在 ICML 2026 提出 **ClimateAR**，首个面向概率气候预测的自回归生成模型[^src-climatear]。

## 核心问题

气候预测（>1 月 lead time）中，短期网格解析信号衰减，主要可预测性来自 ENSO 等内部气候变率模态。现有方法的两大痛点：
1. **异质气象数据**：CMIP6 模拟数据与 ERA5 再分析数据分布差异大，直接迁移性能下降[^src-climatear]。
2. **复杂条件建模**：气候状态信息密度极高，现有生成式 conditioning（如 text prompt）无法有效注入遥相关等跨尺度约束[^src-climatear]。

## 核心贡献

**1. 对齐分词器（Aligned Tokenizer）**：采用 VQ + 分段码本（segmented codebook）将高维气象变量编码为多尺度离散 token；通过浅层分离+深层共享架构实现模拟-真实数据的跨域语义对齐[^src-climatear]。

**2. 混合尺度条件控制（Mixed-Scale Conditioning）**：结合 intra-scale mixed token（尺度内局部引导）与 hybrid-scale prompt（跨尺度全局前缀），同时捕获多尺度气候交互（如 ENSO 对区域温度的调控）[^src-climatear]。

**3. 噪声增强 Teacher-Forcing**：训练时随机替换部分 token 为噪声，缩小训练-推理分布差距，缓解自回归误差累积[^src-climatear]。

## 实验与性能

在 ERA5 + ORAS5 再分析数据（34 变量，1958–2014）上评估 1–10 月 lead time。相比 Pangu/GraphCast/Oneforecast/ClimaX 及 DWD 物理系统，ClimateAR **平均 ACC 提升 37.56%**，在 t2m/t1000 等近地表变量上增益尤为显著[^src-climatear]。ENSO 指数预测展示强概率校准能力，集合预报能捕获观测不确定性[^src-climatear]。

## 局限

迭代预报的累积误差、对极端异常事件的预测敏感性、以及任务扩展（降维分析、异常检测）仍待探索[^src-climatear]。

[^src-climatear]: [[source-climatear]]
