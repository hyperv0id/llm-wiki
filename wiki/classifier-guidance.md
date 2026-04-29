---
title: "分类器引导 (Classifier Guidance)"
type: technique
tags:
  - diffusion
  - conditional-generation
  - guidance
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# 分类器引导

**分类器引导**（Classifier Guidance）是一种用于[[diffusion-model|扩散模型]]条件生成的技术，通过在采样过程中利用预训练分类器的梯度，将生成结果导向特定的目标类别 $y$[^src-understanding-diffusion-models]。

## 定义

分类器引导的核心思想是：在无条件扩散模型的基础上，额外训练一个能够处理任意噪声水平输入的分类器 $p(y | x_t)$，然后在采样时使用该分类器的对数梯度来修正得分函数的估计，使采样轨迹偏向条件分布 $p(x_t | y)$ 而非边缘分布 $p(x_t)$[^src-understanding-diffusion-models]。

## 数学形式

条件得分函数可以通过贝叶斯定理分解为无条件得分与分类器梯度之和：

$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \gamma \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)
$$

其中[^src-understanding-diffusion-models]：
- $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y)$ 是条件得分函数，指导采样生成类别 $y$ 的样本
- $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ 是无条件得分函数，由预训练的扩散模型给出
- $\nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)$ 是分类器的对数梯度，将采样过程拉向类别 $y$
- $\gamma \geq 0$ 是引导强度系数，控制条件约束的影响力

## 工作流程

1. **训练无条件扩散模型**：在训练数据上学习得分函数 $\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$。
2. **训练噪声鲁棒分类器**：在对应扩散过程各噪声水平的加噪数据上训练分类器 $p_\phi(y | \mathbf{x}_t, t)$。
3. **引导采样**：在每一步去噪过程中，将无条件得分与分类器梯度相加：
   $$
   \tilde{\mathbf{s}}_\theta(\mathbf{x}_t, t) = \mathbf{s}_\theta(\mathbf{x}_t, t) + \gamma \nabla_{\mathbf{x}_t} \log p_\phi(y | \mathbf{x}_t, t)
   $$
4. **使用修正后的得分函数**运行朗之万动力学或 DDPM 采样器生成样本。

## 引导强度 $\gamma$ 的作用

$\gamma$ 控制条件生成中**保真度与多样性**之间的权衡[^src-understanding-diffusion-models]：

| $\gamma$ | 效果 |
|----------|------|
| $\gamma = 0$ | 退化为无条件生成，完全忽略类别信息 |
| $\gamma > 0$ | 生成结果更严格遵循目标类别，但样本多样性下降 |
| $\gamma$ 过大 | 模式坍缩风险，生成样本趋同于少数几种模式 |

实际应用中通常取 $\gamma \in [1, 10]$ 之间的值。

## 局限性

1. **额外分类器训练成本**：需要训练一个与扩散模型独立的分类器，该分类器必须能处理扩散过程中任意噪声水平的输入，这增加了训练复杂度和计算开销[^src-understanding-diffusion-models]。
2. **梯度计算负荷**：采样时每一步都需要计算分类器对输入的梯度，显著增加生成时间。
3. **噪声鲁棒性要求**：分类器必须在各种噪声水平上都有准确的梯度，这对分类器的架构和训练提出了特殊要求。

## 后续发展

由于上述局限性，分类器引导在很大程度上被[[classifier-free-guidance|无分类器引导]]（Classifier-Free Guidance, CFG）所取代。CFG 无需额外训练分类器，通过在训练时同时学习条件和无条件模型、采样时对两者进行插值来实现条件控制，已成为扩散模型条件生成的主流方法[^src-understanding-diffusion-models]。

## 引用

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
