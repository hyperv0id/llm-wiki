---
title: "Multi-Scale Linear Prediction"
type: technique
tags:
  - linear-model
  - multi-scale
  - time-series
  - traffic-prediction
  - trend-extraction
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Multi-Scale Linear Prediction

**多尺度线性预测**是一种用于提取时间序列长程趋势模式的技术，在 [[dst-mamba|DST-Mamba]] (AAAI 2025) 的趋势组件中被提出。其核心思想是：通过多级下采样获取不同粒度的趋势序列，再用自上而下混合策略融合跨尺度信息，最后用逐点线性映射生成预测[^src-dst-mamba]。

## 动机

线性模型（如 [[ltsf-linear|DLinear]]）在时间序列预测中展现出惊人的效率，擅长捕获稳定的长期趋势。然而，单尺度线性模型只能感知一种粒度的趋势模式。实际交通数据在不同时间尺度上表现出多层次趋势——细粒度的逐小时变化、中粒度的日周期模式、粗粒度的周趋势——需要多尺度建模[^src-dst-mamba]。

## 技术流程

### 1. 多尺度下采样

对趋势序列 X_TR ∈ R^(L×N)，通过平均池化逐级下采样生成 m 个尺度[^src-dst-mamba]：

```
X_TR_0 = X_TR                  (原始粒度，最精细)
X_TR_1 = AvgPool_2(X_TR_0)     (1/2 长度)
X_TR_2 = AvgPool_2(X_TR_1)     (1/4 长度)
...
X_TR_{m-1} = AvgPool_2(...)    (最粗粒度，最宏观)
```

### 2. 自上而下混合（Top-Down Mixing）

从最粗尺度开始，逐级向下混合[^src-dst-mamba]：

```
X'_{TR_{i}} = X_TR_i + ScaleMix(X_TR_{i+1})
```

其中 ScaleMix 由 MLP 实现，将粗尺度特征上采样以匹配细尺度的时序维度。这种设计确保宏观趋势引导细粒度预测，同时抑制噪声。

### 3. 逐尺度预测与聚合

混合后的各尺度趋势通过独立的线性层映射到对应尺度的预测值，最终聚合为综合趋势预测 Ŷ_TR。

## 与相关技术的对比

| 技术 | 尺度处理 | 混合方向 | 应用 |
|------|---------|---------|------|
| Multi-Scale Linear Prediction | 下采样 + 独立线性 | 自上而下 | 趋势预测（DST-Mamba） |
| [[timemixer|TimeMixer]] PDM | 下采样 + MLP 混合 | 自上而下 | 通用时序预测（季节+趋势） |
| [[autoformer|Autoformer]] 渐进分解 | 单尺度均值滤波 | 无跨尺度交互 | 趋势提取 |

DST-Mamba 的独特之处在于：多尺度仅用于**趋势成分**，季节成分由 Mamba 编码器独立处理——这种分工使两个组件各司其职[^src-dst-mamba]。

## 适用场景

- 趋势-季节分离的时序预测架构
- 需要多粒度趋势感知的长期预测任务
- 计算资源受限场景（线性操作计算开销极低）

[^src-dst-mamba]: [[source-dst-mamba]]
