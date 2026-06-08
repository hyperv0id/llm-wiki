---
title: "Autoregressive Consistency Models"
type: concept
tags:
  - consistency-models
  - autoregressive-generation
  - sequential-modeling
  - weather-forecasting
  - diffusion-models
  - probabilistic-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Autoregressive Consistency Models (自回归一致性模型)

**自回归一致性模型**是将 [[consistency-models|一致性模型]] 的单步生成能力扩展到序列/自回归预测场景的范式，由 [[swift|Swift]] (Stock et al., arXiv 2025) 首次实现于天气预测[^src-swift]。

## 与传统一致性模型的区别

传统的 [[consistency-models|一致性模型]] (ICML 2023) 设计用于独立样本生成（如图像）：从纯噪声 $x_T$ 单步映射到干净数据 $x_0$。每个样本独立生成，无时间依赖[^src-consistency-models]。

自回归一致性模型需要处理序列预测任务：每个预测步骤的输出成为下一步骤的条件输入，形成自回归 rollout。这带来两个关键挑战[^src-swift]：

1. **误差累积**：单步预测的微小误差在自回归 rollout 中指数放大
2. **缺乏不确定性建模**：标准一致性模型训练目标（ℓ₂ 类）不鼓励生成多样化的集合成员

## Swift 的解决方案

### TrigFlow 参数化

Swift 采用 TrigFlow（v-prediction 参数化），将 EDM 和 Flow Matching 统一在共同三角插值轨迹上，简化了一致性参数化[^src-swift]：

$$
f_\theta(x_t, t) = \cos(t)x_t - \sin(t)\sigma_d F_\theta(x_t/\sigma_d, t)
$$

当 $t = \pi/2$ 时，简化为单步预测，每步仅需 1 NFE[^src-swift]。

### CRPS 自回归微调

预训练后，Swift 通过 [[crps-autoregressive-finetuning|CRPS 自回归微调]] 直接优化集合校准度。在多步 rollout 上反向传播 CRPS 损失，迫使模型学会产生校准良好的集合预报[^src-swift]。

### 动态时间间隔

在训练过程中随机采样时间间隔 δi ∼ U{6, 12, 24}，使模型能泛化到不同步长，同时保留时间保真度[^src-swift]。

## 优势

| 维度 | 扩散模型 | 自回归一致性模型 |
|------|---------|----------------|
| NFE/步 | 20–40 | **1** |
| 推理速度 | O(10 min)/预报 | **O(10 s)/预报** |
| 多步微调 | 禁止性昂贵 | 可行（NFE=1） |
| 最长稳定预报 | ~15 天 | **75 天** |

这一效率提升使自回归微调成为可能——与扩散基线（7.6 min/12 预报）相比，Swift 的推理成本（15s/64 预报）降低了 30× 墙钟时间[^src-swift]。

## 应用前景

- **次季节-季节（S2S）预测**：75 天稳定预报覆盖 ENSO 预报的关键尺度
- **集合预报系统**：单模型即可替代 multi-model 集合（GenCast 依赖 4 个独立模型）
- **蒸馏流水线**：从大型扩散模型（如 Aeris）蒸馏到高效一致性模型以降低部署成本[^src-swift]

## 局限

- 仅在一类架构（Swin Transformer）上验证
- 散布-技能权衡：微调改善长期散布但恶化短期 SSR（0–4 天）
- 仅使用 ERA5 数据，未测试其他再分析或观测数据[^src-swift]

## 相关页面

- [[swift]] — Swift 模型
- [[consistency-models]] — 一致性模型基础
- [[crps-autoregressive-finetuning]] — CRPS 微调技术
- [[trigflow]] — TrigFlow 框架
- [[diffusion-models]] — 扩散模型（被加速的基线）
- [[probability-flow-ode]] — 概率流 ODE
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[ensemble-forecasting-calibration]] — 集合预报校准

[^src-swift]: [[source-swift]]
[^src-consistency-models]: [[source-consistency-models]]
