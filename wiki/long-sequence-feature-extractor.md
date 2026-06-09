---
title: "Long Sequence Feature Extractor (LSFE)"
type: technique
tags:
  - time-series
  - traffic-forecasting
  - linear-attention
  - pre-training
  - precomputation
created: 2026-06-09
last_updated: 2026-06-09
source_count: 1
confidence: medium
status: active
---

# Long Sequence Feature Extractor (LSFE)

**长序列特征提取器 (LSFE)** 是 [[bigst|BigST]] 的预处理阶段：把长历史序列（如过去一周数千步）编码成低维表征并**预计算缓存**，从而把长程建模的重计算移出预测阶段[^src-bigst]。

## 两个模块
- **上下文感知线性化 Transformer**：先用膨胀时间卷积提取点级局部上下文（弥补单点语义弱），再用 Performer **正随机特征 (PRF)** 近似 softmax 核 exp(qᵀk)≈φ(q)ᵀφ(k)；因 Σφ(k)v、Σφ(k) 跨查询共享，复杂度从 O(T_l²) 降到 **O(T_l)**；以**生成式预训练**（由长历史预测未来）学习长程时间动态[^src-bigst]。
- **周期特征采样**：免训练，从过去 D 天、W 周的同期区间采样流量特征并聚合为 h_pe，显式表征日/周周期[^src-bigst]。

## 设计意义
核心是**解耦**：长序列建模与空间预测分离，输出 H_long、H_per 可整库预计算缓存，使预测阶段 LGSCN 只需吃最近 T（如 12）步 + 缓存特征，兼顾长程信息与大规模效率[^src-bigst]。消融显示去掉长程表征或周期特征都掉点，弱周期的北京更依赖长程表征[^src-bigst]。

## 关联
- [[bigst]] — 提出 LSFE 的模型
- [[linearized-spatial-convolution]] — BigST 空间维的对偶机制（同用 PRF 核线性化）
- [[informer]]、[[patchtst]] — 高效 / 分块 Transformer 时序建模脉络
- [[traffic-forecasting]] — 任务

[^src-bigst]: [[source-bigst]]
