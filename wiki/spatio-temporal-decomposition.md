---
title: "Spatio-Temporal Decomposition"
type: concept
tags:
  - decomposition
  - spatio-temporal
  - traffic-prediction
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Spatio-Temporal Decomposition

**时空分解**（Spatio-Temporal Decomposition）是指在时空预测任务中将原始交通序列显式拆分为趋势成分和季节成分，分别用适合其统计特性的 backbone 进行建模[^src-dst-mamba]。与纯时序分解不同，时空分解还需考虑路网空间约束对季节波动的影响。

## 与纯时序分解的对比

传统的时序分解方法（如 [[autoformer|Autoformer]] 的渐进式序列分解、[[fedformer|FEDformer]] 的频域增强分解、[[timemixer|TimeMixer]] 的 Past-Decomposable-Mixing）仅关注**时序维度的纠缠**——将时间序列分解为趋势/季节子序列后，仍用统一的 backbone（Transformer 或 MLP）处理所有成分[^src-dst-mamba]。

[[dst-mamba|DST-Mamba]] (AAAI 2025) 首次将分解思想扩展至**时空维度**[^src-dst-mamba]：

```
                             纯时序分解
  X (L × N) ──→ X_TR + X_SE ──→ 统一 backbone ──→ Ŷ

                             时空分解 (DST-Mamba)
  X (L × N) ──→ ┌─ X_TR ──→ 多尺度线性预测 ──→ Ŷ_TR
                 └─ X_SE ──→ 双向 Mamba (空间) ──→ Ŷ_SE
                              └────────┬────────┘
                                    Ŷ = Ŷ_SE + λŶ_TR
```

## 关键洞察

DST-Mamba 的时空分解基于两个观察[^src-dst-mamba]：

1. **趋势成分与节点交互弱相关**：长程趋势（如早晚高峰的宏观走向）在不同节点间表现出高度一致性，无需复杂的空间建模——简单的线性映射即可捕获
2. **季节成分高度依赖空间约束**：短程波动（如局部拥堵传播）受路网拓扑直接影响，需要能捕获跨节点相关性的空间编码器

因此，**趋势用线性 backbone、季节用 Mamba backbone** 是一种基于数据特性的合理分工，而非任意设计选择。

## 相关概念

- [[autoformer|Autoformer]] — 渐进式时序分解 + 自相关机制（NeurIPS 2021）
- [[fedformer|FEDformer]] — 频域增强分解 Transformer（ICML 2022）
- [[timemixer|TimeMixer]] — 多尺度可分解混合（ICLR 2024）
- [[hybrid-periodicity-decoupling|Hybrid Periodicity Decoupling]] — 短周期/长周期混合解耦

[^src-dst-mamba]: [[source-dst-mamba]]
