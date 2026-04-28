---
title: "退火朗之万动力学 (Annealed Langevin Dynamics)"
type: technique
tags:
  - sampling
  - score-based
  - diffusion-models
  - ncsn
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 退火朗之万动力学

**退火朗之万动力学 (Annealed Langevin Dynamics, ALD)** 是 NCSN 论文中提出的采样方法，通过从高噪声水平逐步降低到低噪声水平来改善采样质量[^src-ncsn]。

## 动机

标准朗之万动力学在多峰分布中面临两个问题：
1. **低密度区域分数估计不准确**：缺乏训练信号
2. **模式间混合慢**：需要穿越低密度区域才能到达其他模式

## 算法

```
输入: 噪声水平序列 {σ₁, σ₂, ..., σ_L}, 步长 ε, 迭代数 T
初始化: x̃₀ ~ π (先验分布，如均匀噪声)

for i = 1 to L do
    α_i = ε · σ_i² / σ_L²    # 步长随噪声水平缩放
    for t = 1 to T do
        z_t ~ N(0, I)
        x̃_t = x̃_{t-1} + α_i ∇_x log q_{σ_i}(x̃_{t-1}) + √α_i z_t
    end for
    x̃₀ = x̃_T  # 作为下一阶段的初始样本
end for

return x̃_T
```

## 关键设计

### 噪声序列

使用几何序列 {σᵢ}，满足：
- σ₁ 足够大：填充低密度区域，使模式不那么孤立
- σ_L 足够小：使扰动分布接近原始数据分布

### 步长缩放

α_i ∝ σ_i² 的选择使得信噪比在各噪声水平保持一致：
$$
\frac{E[||\alpha_i \nabla_x \log q_{\sigma_i}||]}{E[||\sqrt{\alpha_i} z||]} \approx \text{constant}
$$

## 为什么有效

1. **高噪声填充低密度区域**：使分数估计更准确
2. **模式间混合改善**：大噪声连接分离的模式
3. **渐进式细化**：每阶段的样本作为下一阶段的好起点

## 与模拟退火的关系

ALD 灵感来自**模拟退火 (Simulated Annealing)**，后者通过逐步降低"温度"参数来优化多峰目标函数。ALD 将这一思想应用于基于分数的采样。

## 后续发展

- **DDPM** 使用类似的退火思想，但通过扩散过程的前向/反向过程实现
- **Score-Based SDE** 将退火推广为连续的随机微分方程

## 相关页面

- [[ncsn]] — NCSN 使用退火朗之万动力学作为采样方法
- [[langevin-dynamics]] — 标准朗之万动力学
- [[score-based-generative-modeling]] — 基于分数的生成建模概念
- [[smld]] — SMLD 框架中的采样方法

## 引用

[^src-ncsn]: [[source-ncsn]]