---
title: "Predictor-Corrector Sampling"
type: technique
tags:
  - diffusion-models
  - sampling
  - sde
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Predictor-Corrector (PC) Sampling

**预测-校正采样器 (Predictor-Corrector Sampler, PC Sampler)** 是 Score-Based SDE 论文提出的采样框架，结合数值 SDE 求解器与基于分数的 MCMC 方法来提高采样质量[^src-sde]。

## 核心思想

PC 采样器在每一步迭代中交替使用：
- **Predictor（预测器）**：数值 SDE 求解器，给出下一步的估计
- **Corrector（校正器）**：基于分数的 MCMC 方法，修正边缘分布

## 算法流程

```
for t = T to 0:
    # Predictor: 数值 SDE 求解器
    x_t-1 = solver_step(s_θ, x_t, t)
    
    # Corrector: Langevin MCMC 或 HMC
    x_t-1 = corrector_step(s_θ, x_t-1, t)
```

## 与传统方法的关系

| 方法 | Predictor | Corrector |
|------|-----------|-----------|
| SMLD (原始) | 恒等函数 | 退火 Langevin |
| DDPM (原始) | 祖先采样 | 恒等函数 |
| **PC 采样器** | 任意 SDE 求解器 | 任意分数 MCMC |

## 优势

1. **通用性**：可应用于任何 SDE 变体（VE, VP, sub-VP）
2. **质量提升**：比单纯增加采样步数效果更好
3. **兼容性**：可与离散训练的模型配合使用

## 实验结果

在 CIFAR-10 上，PC 采样器相比纯预测器显著提升 FID：
- P1000 (仅预测器): FID 3.24
- PC1000 (预测+校正): FID 2.92

## 引用

[^src-sde]: [[source-sde]]