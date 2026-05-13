---
title: "Projected Fisher Divergence"
type: technique
tags:
  - divergence
  - score-matching
  - flux-matching
  - fisher-divergence
  - generative-model
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: medium
status: active
---

# Projected Fisher Divergence

**投影 Fisher 散度**（Projected Fisher Divergence）是 Flux Matching 中定义的一种统计散度，用于在学习分布保持向量场时保持 Fisher 散度的 $L^2(p_{\text{data}})$ 几何。[^src-2605-07319]

## 定义

设 $\Pi_{\text{flux}}$ 为将任意向量场投影到与其具有相同概率通量散度的梯度场的算子。投影 Fisher 散度为：

$$
\tilde{J}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \|\Pi_{\text{flux}} f_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2
$$

其中 $f_\theta$ 是待学习的向量场，$\nabla \log p_{\text{data}}$ 是得分函数。[^src-2605-07319]

## 关键性质

1. **分布保持不变性**：对任何满足 $\nabla \cdot (p_{\text{data}} v) = 0$ 的扰动 $v$，有 $\tilde{J}(\theta + v) = \tilde{J}(\theta)$，因为 $\Pi_{\text{flux}}(f_\theta + v) = \Pi_{\text{flux}} f_\theta$。[^src-2605-07319]

2. **与 Fisher 散度同几何**：投影算子 $\Pi_{\text{flux}}$ 确保散度保持在向量场的 $L^2(p_{\text{data}})$ 空间中，避免导数级噪声放大的问题。[^src-2605-07319]

3. **通量条件**：$\Pi_{\text{flux}} f_\theta$ 与 $\nabla \log p_{\text{data}}$ 的投影 Fisher 散度零值等价于 $f_\theta$ 驱动的扩散以 $p_{\text{data}}$ 为平稳分布。[^src-2605-07319]

## 与 Fisher 散度的关系

Fisher 散度 $J(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \|f_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2$ 要求逐点匹配得分，仅对得分函数本身零值。投影 Fisher 散度放松了这一条件——它对所有满足 Fokker-Planck 平稳条件的生成向量场零值。[^src-2605-07319]

由于 $\nabla \log p_{\text{data}}$ 本身是梯度场且 $\Pi_{\text{flux}}(\nabla \log p_{\text{data}}) = \nabla \log p_{\text{data}}$，当 $f_\theta = \nabla \log p_{\text{data}}$ 时两者等价。

## 计算

直接计算投影 Fisher 散度在高维空间不可行。Flux Matching 通过证明 $\nabla_\theta \tilde{J}(\theta) = 2 \nabla_\theta L_{\text{flux}}(\theta)$（定理 3.1），将优化转化为可计算的损失 $L_{\text{flux}}$。[^src-2605-07319]

## 相关页面

- [[flux-matching]] — 使用投影 Fisher 散度的生成建模范式
- [[score-matching]] — 传统 Fisher 散度的应用
- [[fokker-planck-equation]] — 福克-普朗克方程的平稳条件

## 引用

[^src-2605-07319]: [[source-2605-07319]]
