---
title: "缩放因子 1/√dₖ"
type: concept
tags:
  - transformer
  - attention
  - numerical-stability
  - softmax
  - variance-control
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: medium
status: active
---

# 缩放因子 $1/\sqrt{d_k}$

缩放因子 $1/\sqrt{d_k}$ 是 Scaled Dot-Product Attention 中的核心数值稳定性设计，在点积结果进入 Softmax 之前将其方差从 $d_k$ 归一化至 1，防止高维条件下 Softmax 饱和与梯度消失[^src-bluuuuue-scaling-factor-intuition]。

## 数学原理

### 点积方差膨胀

设 Query 向量 $Q$ 与 Key 向量 $K$ 各分量独立且服从 $N(0,1)$。点积 $Z = \sum_{i=1}^{d_k} q_i k_i$ 涉及 $d_k$ 个乘积项求和。每个乘积项 $q_i k_i$ 的方差为 $Var(q_i) \cdot Var(k_i) = 1$（利用独立零均值变量的乘积方差性质）。因此[^src-bluuuuue-scaling-factor-intuition]：

$$Var(Z) = d_k, \quad \sigma_Z = \sqrt{d_k}$$

$d_k$ 越大，点积值的分布越分散。当 $d_k = 64$ 时，点积的典型取值范围已在正负数十的量级[^src-bluuuuue-scaling-factor-intuition]。

### Softmax 饱与梯度消失

Softmax 函数 $\sigma(\mathbf{z})_i = e^{z_i} / \sum e^{z_j}$ 的偏导数为[^src-bluuuuue-scaling-factor-intuition]：

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

当输入方差极大时，最大分量经指数运算占据绝对主导，输出趋近 One-hot。此时 $\sigma_i(1 - \sigma_i) \to 0$，梯度消失——参数 $W_Q$ 和 $W_K$ 无法通过误差信号校正[^src-bluuuuue-scaling-factor-intuition]。

饱和导致的双重后果：
1. **信息表征层面**：注意力退化为单点聚焦，丧失多位置关注能力
2. **梯度传导层面**：Softmax 偏导数趋近于零，注意力头不再接收有效学习信号[^src-bluuuuue-scaling-factor-intuition]

### 缩放的数学推导

将点积除以 $\sqrt{d_k}$ 后[^src-bluuuuue-scaling-factor-intuition]：

$$Var\left(\frac{Z}{\sqrt{d_k}}\right) = \frac{1}{d_k} Var(Z) = \frac{1}{d_k} \cdot d_k = 1$$

方差归一化后，点积值约 95% 落在 $[-2, 2]$ 范围内，Softmax 具有足够的非线性区分度且避开饱和区[^src-bluuuuue-scaling-factor-intuition]。

## 三个数学特性

### 1. 恰好 $1/\sqrt{d_k}$ 而非 $1/d_k$

若除以 $d_k$，方差变为 $d_k \times (1/d_k)^2 = 1/d_k$，标准差 $\sqrt{1/d_k}$。高维条件下方差趋于零——点积被过度压缩，Softmax 输出趋于均匀分布，注意力失去聚焦能力。$1/\sqrt{d_k}$ 恰好将方差稳定在 1，避免了**欠缩放**与**过缩放**两种失效模式[^src-bluuuuue-scaling-factor-intuition]。

### 2. 保持相对排序

缩放操作 $f(x) = x/c$ 是单调递增函数，argmax 运算结果具有缩放不变性。唯一改变的是概率分布在 One-hot 与均匀分布之间的位置——属于纯数值稳定性修正，而非表征层面的改变[^src-bluuuuue-scaling-factor-intuition]。

### 3. 初始化阶段的不可替代性

在训练初期模型尚未学会调整 $Q$ 和 $K$ 的范数时，缩放因子防止因纯粹的数值尺度问题陷入不可逆的饱和状态。层归一化与残差连接使每层注意力输入的统计特性在训练中保持相对稳定，因此缩放因子对深层结构中的注意力稳定性具有持续贡献[^src-bluuuuue-scaling-factor-intuition]。

## 在后续架构中的保留

该缩放因子在 VLA（视觉-语言-动作模型）和世界模型等后续架构的注意力模块中被完整保留——无论是 Cross-Attention 还是 Causal Attention[^src-bluuuuue-scaling-factor-intuition]。其不可替代性来自所解决问题的结构性质：只要注意力沿用"高维向量内积施加 Softmax"这一基本形式，点积方差膨胀就是必然发生的数学事实。

## 与相关概念的关系

| 概念 | 与缩放因子的关系 |
|------|-----------------|
| [[attention-entropy-collapse]] | 缩放因子防止方差膨胀导致的熵崩溃，是根本性的预防机制 |
| [[attention-logit-explosion]] | 缩放因子控制初始化阶段的 logit 尺度，防止 Q/K 范数增长的数值后果 |
| [[attention-temperature-scaling]] | 温度缩放是 $1/\sqrt{d_k}$ 的推广——在 RoPE 上下文扩展中调节 $t$ 值 |
| [[key-normalization]] | 键归一化从另一角度解决注意力不稳定——约束 K 范数而非缩放点积 |
| [[entropy-boundedness]] | 缩放因子确保注意力熵远离零（防止饱和）和远离最大值（防止均匀化） |

## 引用

[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]
