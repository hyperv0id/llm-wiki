---
title: "Generative Vector Field"
type: concept
tags:
  - generative-model
  - vector-field
  - fokker-planck
  - diffusion-models
  - flux-matching
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: medium
status: active
---

# Generative Vector Field

**生成向量场**（Generative Vector Field）是指驱动扩散过程后以目标数据分布 $p_{\text{data}}$ 为平稳分布的向量场。根据福克-普朗克方程，有无限多个生成向量场共享同一平稳分布。[^src-2605-07319]

## 形式化定义

考虑扩散过程 $dx_t = f_\theta(x_t) dt + \sqrt{2} dW_t$，其边际密度 $p_t$ 的演化由福克-普朗克方程描述：

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (p_t(x) f_\theta(x)) + \nabla \cdot (p_t(x) \nabla \log p_t(x))
$$

在平稳状态下 $\partial_t p_t = 0$，$p_{\text{data}}$ 是平稳分布的充要条件为：

$$
\nabla \cdot (p_{\text{data}}(x) f_\theta(x)) = \nabla \cdot (p_{\text{data}}(x) \nabla \log p_{\text{data}}(x))
$$

所有满足上述条件的 $f_\theta$ 称为 $p_{\text{data}}$ 的生成向量场。[^src-2605-07319]

## 自由度

生成向量场可参数化为：

$$
f_\theta(x) = \nabla \log p_{\text{data}}(x) + v(x), \quad \nabla \cdot (p_{\text{data}}(x) v(x)) = 0
$$

其中 $\nabla \log p_{\text{data}}$ 是唯一的保守（梯度）部分（得分函数），$v(x)$ 是任意无散度扰动（相对于权重 $p_{\text{data}}$）。正是 $v(x)$ 的自由度使得生成向量场可以编码额外的动力学属性。[^src-2605-07319]

## 与得分函数的区别

| 性质 | 得分函数 $\nabla \log p_{\text{data}}$ | 一般生成向量场 |
|------|--------------------------------------|---------------|
| Jacobian | 对称（保守场） | 可不对称 |
| 混合速度 | 慢（可逆 Langevin） | 可优化加速 |
| 有向依赖 | 无法编码 | 可编码 |
| 唯一性 | 唯一 | 无限多 |

## 应用价值

- **加速采样**：优化混合速度减少 Langevin 步数
- **可解释建模**：使用具有科学意义的参数化形式（如 RNA 速度的机制性 ODE）
- **结构先验**：通过因果掩码编码有向依赖关系
- **非保守动力学**：学习具有环流（circulation）等非保守性质的向量场[^src-2605-07319]

## 相关页面

- [[flux-matching]] — 学习生成向量场的方法
- [[projected-fisher-divergence]] — 用于学习生成向量场的散度
- [[fokker-planck-equation]] — 生成向量场分类的物理基础
- [[score-function]] — 得分函数，生成向量场的特例
- [[score-based-generative-modeling]] — 基于分数的生成建模

## 引用

[^src-2605-07319]: [[source-2605-07319]]
