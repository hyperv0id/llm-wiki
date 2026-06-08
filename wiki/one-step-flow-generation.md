---
title: "One-Step Flow Generation"
type: technique
tags:
  - flow-matching
  - generative-model
  - efficient-inference
  - one-step-generation
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# One-Step Flow Generation (一步流生成)

## 定义

**一步流生成**是指通过学习直线（甚至恒定速度）概率路径，使流匹配模型在仅需单次函数求值（NFE=1）的情况下完成从噪声到数据的高质量生成的技术[^src-cogencast]。区别于需要多步 ODE 求解的标准流匹配（[[sundial|Sundial]] 需多步采样、[[flowts|FlowTS]] 需 30 步）和需要迭代去噪的扩散模型。

## 理论基础

### 为什么标准流匹配需要多步？

标准流匹配学习的是**瞬时**向量场 $v_t(x)$，需通过 ODE 求解器从 $t=0$ 积分到 $t=1$：

$$
x_1 = x_0 + \int_0^1 v_t(x_t) dt
$$

即使学习直线 OT 路径，数值积分仍需要离散化步骤（通常 5-100 步）以保证精度。

### CoGenCast 的方案：平均速度建模

CoGenCast 的 [[average-velocity-modeling|平均速度建模]] 是使一步生成可行的核心技术[^src-cogencast]。不同于预测瞬时向量场，CoGenCast 学习**区间条件化的平均速度** —— 给定当前时间 $t$ 和目标时间 $r$（$r > t$），直接预测从 $t$ 到 $r$ 所需的平均速度：

$$
v_{\text{avg}}([t, r]) \approx \frac{1}{r-t} \int_t^r v_\tau d\tau
$$

训练时在 $r=1$ 处采样，使得模型能直接从噪声（$t=0$）一步预测到干净数据（$r=1$）[^src-cogencast]。

### JVP 修正的优化目标

为使平均速度场真正趋近恒定速度（即直线轨迹），CoGenCast 采用 **Jacobian-Vector Product (JVP)** 修正的损失函数[^src-cogencast]：

$$
\mathcal{L} = \mathbb{E}_{t,r,\epsilon,y}\left[\frac{1}{N}\sum_{j=1}^N \left\| u_j^{\text{out}} - v_j - (r-t)\frac{\partial u_j^{\text{out}}}{\partial t} \right\|_2^2\right]
$$

其中 $\frac{\partial u_j^{\text{out}}}{\partial t}$ 是速度场的**时间偏导数**，通过 JVP 高效计算。$(r-t)\frac{\partial u}{\partial t}$ 项是 Taylor 展开的一阶修正，显式惩罚速度变异性，驱动物理轨迹趋近直线[^src-cogencast]。

## 一步生成的实现

### 训练

1. 采样 $t \sim \mathcal{U}[0, 1]$，$r \sim \mathcal{U}[t, 1]$
2. 构造插值：$\hat{y} = (1-t)\epsilon + t y$（线性 OT 路径）
3. 去噪 decoder 预测平均速度 $u^{\text{out}}$
4. 计算 JVP 修正后的 MSE 损失

### 推理

对于每个 patch，仅需**一次前向传播**[^src-cogencast]：

1. 采样纯高斯噪声 $y^{(0)} \sim \mathcal{N}(0, I)$（对应 $t=0$）
2. 前向传播得到平均速度场 $u$
3. 一步积分恢复：$y^{\text{out}} = y^{(0)} + u$

### Linear Scheduler 的关键作用

Linear noise scheduler 与平均速度建模天然对齐[^src-cogencast]：
- **Linear**：均匀离散化，配合恒定速度假设，一步生成精度最优
- **Cosine**：引入不必要的时间曲率，破坏直线轨迹假设，性能显著下降

## 实验验证

CoGenCast 的 NFE 消融实验证明[^src-cogencast]：
- **NFE=1** 已取得最优或接近最优的预测精度
- **NFE=2-3** 仅在少数特定场景有边际增益
- 这验证了学习到的流轨迹确实高度直线化

## 与其他少步生成方法的对比

| 方法 | 技术路线 | 最少步数 | 领域 |
|------|---------|---------|------|
| **CoGenCast** | 平均速度 + JVP 修正 | **1** | 时间序列 |
| [[instaflow|InstaFlow]] | Reflow + 蒸馏 | 1 | 图像 |
| [[consistency-models|Consistency Models]] | 自洽性映射 | 1-2 | 图像 |
| [[shortcut-models|Shortcut Models]] | 自洽性 + 步长条件化 | 1 | 图像 |
| [[rectified-flow|Rectified Flow]] | Reflow 迭代拉直 | 2-4 | 图像 |
| [[flowts|FlowTS]] | Rectified Flow 原生 | 30 | 时间序列 |

## 相关页面

- [[cogencast]] — CoGenCast，首个一步流生成时间序列模型
- [[average-velocity-modeling]] — 平均速度建模，一步生成的理论基础
- [[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-FM 预测范式
- [[flow-matching]] — 流匹配理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测全景

[^src-cogencast]: [[source-cogencast]]