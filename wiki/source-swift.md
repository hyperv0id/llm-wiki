---
title: "Swift — An Autoregressive Consistency Model for Efficient Weather Forecasting"
type: source-summary
tags:
  - weather-forecasting
  - consistency-models
  - diffusion-models
  - probabilistic-forecasting
  - ensemble
  - arxiv-2025
  - generative-time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: medium
status: active
---

# Source: Swift (arXiv 2025)

Swift (Stock et al., Argonne National Laboratory, arXiv:2509.25631, Sep 2025) proposes the **first autoregressive consistency model for probabilistic weather forecasting**, achieving 39× faster inference than diffusion baselines with 75-day stable forecasts[^src-swift].

## 核心贡献

1. **Temporal Consistency Model**：首次将连续时间 [[consistency-models|Consistency Model]] 应用于天气预测自回归 rollout，单步 NFE=1 替代扩散模型的 20–40 NFE。基于 TrigFlow 统一 EDM 和 Flow Matching 的 v-prediction 参数化[^src-swift]。

2. **CRPS Autoregressive Finetuning**：提出通过 CRPS（连续秩概率得分）在多步自回归 rollout 上微调一致性模型，使原本缺乏不确定性建模的一致性模型能产生校准良好的集合预报[^src-swift]。

3. **Swin Transformer 架构**：采用 225M 参数 conditional non-hierarchical Swin Transformer，具有移位窗口、adaLN 调制和动态时间间隔 δi ∼ U{6, 12, 24}[^src-swift]。

## 训练

两阶段训练：(1) 预训练：15M images，TrigFlow v-prediction 目标 + Muon 优化器；(2) 多步微调：5M images，CRPS 损失在 K=1–8 自回归步数上通过序列梯度检查点反向传播[^src-swift]。

## 结果

- **39× 加速**：1 NFE/步 vs 扩散基线 39 NFE[^src-swift]
- **75 天稳定预报**：超越次季节-季节尺度（S2S）的稳定性要求[^src-swift]
- **集合校准**：在 Hurricane Laura 案例中生成多样且真实的风暴轨迹[^src-swift]
- **竞争 IFS ENS**：中程预报技能与数值集合系统竞争[^src-swift]

## 局限

预报散布不足（SSR < 1），极地区域误差较大，平滑场（位势高度、海平面气压）在高纬向波数上有谱漂移[^src-swift]。代码和权重开源：https://github.com/stockeh/swift。

[^src-swift]: [[source-swift]]
