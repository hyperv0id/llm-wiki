---
title: "得分匹配 (Score Matching)"
type: concept
tags:
  - generative-models
  - estimation
  - diffusion-models
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 得分匹配

**得分匹配** 是 Aapo Hyvärinen 于 2005 年提出的非归一化统计模型估计方法。它通过匹配对数密度梯度（得分函数）来进行参数估计，无需计算归一化常数。[^src-tutorial]

## 定义

数据分布 $p(x)$ 的**得分函数**定义为：

$$
\nabla_x \log p(x)
$$

得分匹配的训练目标是最小化模型得分与真实得分之间的 Fisher 散度：

$$
\frac{1}{2} \mathbb{E}_{p(x)} \left[ \| s_\theta(x) - \nabla_x \log p(x) \|^2 \right]
$$

## 显式得分匹配的问题

直接优化需要计算 Hessian 矩阵的对角线元素 $\nabla_x^2 \log p(x)$，计算量巨大，无法扩展到高维数据。[^src-tutorial]

## 去噪得分匹配

**去噪得分匹配 (DSM)** 通过扰动数据来规避 Hessian 计算：用 $q(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$ 替换原始分布，训练目标变为：

$$
\frac{1}{2} \mathbb{E}_{p(x)}\mathbb{E}_{q(\tilde{x}|x)} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}|x) \|^2 \right]
$$

其中 $\nabla_{\tilde{x}} \log q(\tilde{x}|x) = (x - \tilde{x}) / \sigma^2$。[^src-tutorial]

**关键结论**：在 $q(\tilde{x}|x)$ 为高斯噪声的条件下，去噪得分匹配等价于训练一个去噪自动编码器。[^src-tutorial]

## 在扩散模型中的应用

- **SMLD**：使用 [[ncsn]] 网络在多个噪声水平上训练得分函数
- **DDPM**：等价于在特定噪声调度下进行得分匹配

## 相关页面

- [[ncsn]] — NCSN 使用分数匹配训练条件分数网络
- [[smld]] — SMLD 框架
- [[annealed-langevin-dynamics]] — 退火朗之万动力学

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
