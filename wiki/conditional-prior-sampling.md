---
title: "Conditional Prior Sampling"
type: technique
tags:
  - flow-matching
  - langevin-dynamics
  - prior-conditioning
  - time-series
  - probabilistic-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Conditional Prior Sampling

**条件先验采样** (Conditional Prior Sampling, CPS) 是一种推理时技术，允许无条件训练的流匹配模型执行条件预测[^src-tsflow]。由 TSFlow (ICLR 2025) 提出[^src-tsflow]。

## 问题

无条件模型 $p_\theta(y)$ 在训练时不知道任何条件信息（如观测到的历史 $y^p$）。如何在推理时让它执行条件预测 $p_\theta(y \mid y^p)$？

## 两步法

TSFlow 将条件预测分解为两步[^src-tsflow]：

### 第一步：条件化先验 (CPS)

从条件先验分布 $q_0(x_0 \mid y^p)$ 采样，使用 **Langevin 动力学**[^src-tsflow]：

$$
x_0^{(i+1)} = x_0^{(i)} - \eta \nabla_{x_0} \log q_0(x_0^{(i)} \mid y^p) + \sqrt{2\eta}\xi_i, \quad \xi_i \sim \mathcal{N}(0, I)
$$

条件得分函数通过贝叶斯规则分解[^src-tsflow]：

$$
\nabla_{x_0} \log q_0(x_0 \mid y^p) = \underbrace{\nabla_{x_0} \log q_1(y^p \mid x_0)}_{\text{引导观测对齐}} + \underbrace{\nabla_{x_0} \log q_0(x_0)}_{\text{保持先验流形}}
$$

其中 $q_1(y^p \mid x_0)$ 建模为非对称拉普拉斯分布 (ALD)，中心位于流输出 $\phi_{\theta,1}(x_0)$ 处[^src-tsflow]。

### 第二步：引导生成

在条件化先验后，通过修改向量场进行**引导生成**[^src-tsflow]：

$$
\tilde{u}_\theta(t, x_t) = u_\theta(t, x_t) - s\sigma_t \nabla_{x_t} \log p_t(y^p \mid x_t)
$$

其中 $s$ 是引导强度参数（通常 8-32，在验证集上选择）[^src-tsflow]。

## 实现细节

- Langevin 迭代次数：通常 4 次，步长 $\eta = 0.005$，噪声尺度 0.5[^src-tsflow]
- 得分函数中的 $\phi_{\theta,1}$ 使用少数 Euler 步近似以加速[^src-tsflow]
- 量化参数 $\kappa \sim \mathcal{U}[0.1, 0.9]$ 确保覆盖不同分位数[^src-tsflow]

## 效果

- CPS + 引导在 7/8 数据集上达到或超越仅使用引导的模型[^src-tsflow]
- TSFlow-Uncond. 在 Electric、Traffic、UberTLC 上与条件版本差距较小（CRPS 0.049 vs 0.045 等）[^src-tsflow]
- 无条件模型 + CPS 的推理速度约是条件模型的 3 倍（更长的上下文窗口 + ODE 微分），但仍远快于扩散模型[^src-tsflow]

## 相关页面

- [[tsflow]] — TSFlow 模型
- [[flow-matching]] — Flow Matching 框架
- [[guided-generation]] — 引导生成技术
- [[langevin-dynamics]] — Langevin 动力学采样
- [[asymmetric-laplace-distribution]] — 非对称拉普拉斯分布

[^src-tsflow]: [[source-tsflow]]
