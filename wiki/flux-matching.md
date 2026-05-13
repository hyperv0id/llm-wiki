---
title: "Flux Matching"
type: concept
tags:
  - generative-model
  - flux-matching
  - fokker-planck
  - score-matching
  - diffusion-models
  - stanford
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: medium
status: active
---

# Flux Matching

**Flux Matching** 是 Peter Pao-Huang、Xiaojie Qiu 和 Stefano Ermon（Stanford University）于 2026 年提出的生成建模范式，推广了基于分数的生成模型（score-based models），将可学习的向量场从唯一的得分函数扩展到无限多个满足福克-普朗克方程平稳性条件的**生成向量场**。[^src-2605-07319]

## 动机

现代生成模型（扩散模型、得分匹配）通常以 Stein 得分函数 $\nabla \log p_{\text{data}}$ 为目标进行学习。然而，福克-普朗克方程表明：**有无限多个向量场与得分函数共享同一平稳分布**。这些非得分向量场可以编码有用的属性——如更快的混合速度、有向依赖关系、机制性结构——而得分匹配会惩罚任何偏离得分的扰动，即使它不改变分布。[^src-2605-07319]

Flux Matching 的核心洞察是：只需要匹配概率通量的散度（而非逐点匹配向量场），就足以保证目标分布不变，同时保留无限多个非得分自由度。

## 理论

### 福克-普朗克条件

考虑扩散过程 $dx_t = f_\theta(x_t) dt + \sqrt{2} dW_t$，其平稳分布为 $p_{\text{data}}$ 的充要条件是：

$$
\nabla \cdot (p_{\text{data}}(x) f_\theta(x)) = \nabla \cdot (p_{\text{data}}(x) \nabla \log p_{\text{data}}(x))
$$

这是 Flux Matching 的理论基础。该条件允许 $f_\theta$ 偏离得分，只要偏离场 $v(x) = f_\theta(x) - \nabla \log p_{\text{data}}(x)$ 满足 $\nabla \cdot (p_{\text{data}}(x) v(x)) = 0$。[^src-2605-07319]

### 投影 Fisher 散度

为在 $L^2(p_{\text{data}})$ 几何中匹配通量散度，论文定义了投影算子 $\Pi_{\text{flux}}$，将任意向量场投影到其产生相同通量散度的梯度场上，对应的投影 Fisher 散度为：

$$
\tilde{J}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \|\Pi_{\text{flux}} f_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2
$$

### Flux Matching 损失

通过 Langevin 扩散的路径积分推导，得到可计算的训练损失：

$$
L_{\text{flux}}(\theta) = -\mathbb{E}_{t \sim q, x_0 \sim p_{\text{data}}, x_t | x_0} \left[ u_\theta(x_0)^\top \cdot \text{sg}\left( \frac{\partial x_t}{\partial x_0^\top} \nabla_{x_t} r_\theta(x_t) \right) \cdot \frac{1}{q(t)} \right]
$$

其中 $u_\theta = f_\theta - \nabla \log p_{\text{data}}$，$r_\theta = p_{\text{data}}^{-1} \nabla \cdot (p_{\text{data}} u_\theta)$，sg 表示 stop-gradient。[^src-2605-07319]

### 噪声退火扩展

Flux Matching 可自然地扩展到噪声退火（noise annealed）设置，在每个噪声水平 $\sigma$ 上独立应用 Flux Matching 损失，学习一个噪声条件向量场 $f_\theta^\sigma(x)$。[^src-2605-07319]

## 关键特点

### 训练与采样解耦

Flux Matching 的损失函数仅用于训练。推理时，学到的向量场可直接替换标准扩散采样器中的得分项——无需任何算法修改。[^src-2605-07319]

### 灵活性

Flux Matching 提供两种策略选取有用的生成向量场：
1. **应用特定损失**：添加正则化项（如 $L_2$ 范数惩罚可恢复得分函数）
2. **模型参数化**：通过架构设计（如注意力掩码）编码先验知识[^src-2605-07319]

## 应用

| 应用领域 | 方法 | 效果 |
|---------|------|------|
| 加速采样 | 添加混合代理损失优化混合速度 | CIFAR-10 上大幅减少采样步数 |
| RNA 速度拟合 | 用 Flux Matching 直接训练 scVelo 参数模型 | 5 个数据集上 4/5 CBC 提升，全部一致性提升 |
| 有向依赖生成 | 因果注意力掩码编码自回归结构 | 轨迹生成中因果掩码始终改善 |
| 独立生成 | 标准 UNet + Flux Matching 损失 | CIFAR-10 FID 9.06，CelebA FID 7.07 |

## 局限

- 训练成本：比 DSM 慢 3-4×，内存多 2-3×
- 依赖 KDE 近似得分，低密度区域效率低
- 超参数敏感（KDE 带宽 $\sigma$、模拟时间 $T$、MCMC 步数）
- 首次实现在图像 FID 上尚未达到 DSM 水平

## 与其他方法的关系

- **Score Matching**：Flux Matching 是 score matching 的推广——允许学习任何分布保持向量场而非唯一的得分函数
- **Flow Matching**：不同范式。Flow Matching 学习生成概率路径的向量场；Flux Matching 学习具有正确平稳分布的扩散漂移场
- **FP-Diffusion**：FP-Diffusion 在扩散训练中强制 FPE 约束；Flux Matching 以 FPE 平稳条件为设计目标

## 相关页面

- [[projected-fisher-divergence]] — Flux Matching 使用的统计散度
- [[score-matching]] — 得分匹配，Flux Matching 推广的目标
- [[fokker-planck-equation]] — 福克-普朗克方程，Flux Matching 的理论基础
- [[flow-matching]] — 流匹配，不同的生成框架
- [[score-based-generative-modeling]] — 基于分数的生成建模
- [[score-function]] — 得分函数

## 引用

[^src-2605-07319]: [[source-2605-07319]]
