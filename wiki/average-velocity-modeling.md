---
title: "Average Velocity Modeling"
type: technique
tags:
  - flow-matching
  - one-step-generation
  - jvp
  - optimal-transport
  - time-series
created: 2026-06-08
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Average Velocity Modeling (平均速度建模)

## 定义

**平均速度建模**是 [[cogencast|CoGenCast]] (ICML 2026) 提出的流匹配训练技术，通过预测区间条件化的平均速度而非瞬时速度，配合 JVP (Jacobian-Vector Product) 修正，使模型能够学习近似直线轨迹，从而实现 [[one-step-flow-generation|一步流生成]][^src-cogencast]。

## 核心洞察

### 问题：瞬时速度场 → 多步积分

标准流匹配学习的是**瞬时**向量场 $v_t(x_t)$，从 $t=0$ 到 $t=1$ 的生成需要通过 ODE 求解器离散积分，需要多步函数求值（NFEs）[^src-cogencast]。

### 方案：区间平均速度 → 一步生成

CoGenCast 转而学习**区间条件化的平均速度**：给定区间 $[t, r]$（$r > t$），直接预测跨越该区间所需的平均速度向量[^src-cogencast]：

$$
u_j^{\text{out}} = \text{DenoisingDecoder}\left(z_j^{\text{in}}, t, r, z_{1:j}^{\text{dec out}}\right)
$$

其中 $z_j^{\text{in}}$ 是加噪 patch 的潜在嵌入，$z_{1:j}^{\text{dec out}}$ 是 LLM decoder 的自回归上下文表示。

### 生成路径：线性插值

从噪声到数据的概率路径采用线性插值[^src-cogencast]：

$$
\hat{y}_{1:N} = (1-t)\epsilon + t y_{1:N}, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这定义了一条 `base velocity direction` $v = y_{1:N} - \epsilon$ 为常数的直线轨迹，为一步生成提供了几何基础。

## JVP 修正的优化目标

### 为什么需要修正？

即使学习平均速度，如果速度场存在时间曲率（即速度随时间非线性变化），一次前向传播仍然无法准确实现完整传输[^src-cogencast]。

### JVP 修正损失

CoGenCast 通过一阶 Taylor 展开修正目标速度[^src-cogencast]：

$$
\mathcal{L} = \mathbb{E}_{t,r,\epsilon,y} \left[ \frac{1}{N}\sum_{j=1}^N \left\| u_j^{\text{out}} - v_j - (r-t)\frac{\partial u_j^{\text{out}}}{\partial t} \right\|_2^2 \right]
$$

其中：
- $u_j^{\text{out}}$ — 模型预测的 patch $j$ 的速度
- $v_j = y_j - \epsilon_j$ — 真实常数速度方向（线性插值的 base velocity）
- $\frac{\partial u_j^{\text{out}}}{\partial t}$ — 速度场的**时间偏导数**，通过 **JVP** 高效计算
- $(r-t)\frac{\partial u_j^{\text{out}}}{\partial t}$ — Taylor 展开的一阶修正项

### 直观理解

$$
\begin{aligned}
\text{目标速度} &= \text{常数速度} + \text{一阶曲率修正} \\
v_{\text{target}} &= v_{\text{constant}} + (r-t) \cdot \frac{dv}{dt}
\end{aligned}
$$

最小化此损失显式惩罚速度的时间变异性，迫使模型学习**近似恒速轨迹**。当 $\frac{\partial u}{\partial t} \to 0$，模型收敛到纯直线[^src-cogencast]。

## 一步推理

利用平均速度的一步推理公式（ODE 积分的离散化近似）[^src-cogencast]：

$$
y_j^{\text{out}} = y_j^{(0)} + \int_0^1 u(z_\tau, \tau, z_{1:j}^{\text{dec out}}) \, d\tau \approx y_j^{(0)} + u_{\text{avg}}
$$

当 NFE=1 时，直接使用模型预测的平均速度作为积分近似。这等价于假设 $u(\tau)$ 在 $[0,1]$ 上近似常数。

## Linear Scheduler 的对齐

Linear noise scheduler 与平均速度建模天然配合[^src-cogencast]：
- 均匀时间离散化 → 每个时间步等权重
- 直线轨迹假设 → 速度场的均匀分布
- Cosine scheduler 引入曲率 → 破坏对齐

实验证实 Linear scheduler 在几乎所有数据集上优于 Cosine[^src-cogencast]。

## 与 MeanFlow 的关系

CoGenCast 的 JVP 修正目标与 [[meanflow|MeanFlow]]（Geng et al., arXiv:2505.13447）的训练目标同构：均以区间条件化网络预测平均速度，并用 $(t-r)$ 量级的一阶导数修正项惩罚速度场时间变化（CoGenCast 记为 $(r-t)\,\partial u/\partial t$，MeanFlow 记为 $(t-r)\,du/dt$，符号差异源于区间方向约定）[^src-cogencast][^src-alphaflow]。[[alphaflow|α-Flow]] 进一步把该目标分解为轨迹流匹配与轨迹一致性两项，报告其梯度强负相关并以课程退火分离两者；图像生成侧的消融显示先充分流匹配预训练再过渡到平均速度目标收敛更好[^src-alphaflow]。

## 相关页面

- [[cogencast]] — 首个应用平均速度建模的模型
- [[meanflow]] — 同构目标的图像生成侧框架
- [[alphaflow]] — 对 MeanFlow 目标的分解分析与课程退火改进
- [[one-step-flow-generation]] — 一步流生成的技术全景
- [[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-FM 预测范式
- [[flow-matching]] — 流匹配理论基础（学习瞬时速度场）
- [[rectified-flow]] — Rectified Flow，通过 reflow 迭代拉直轨迹

[^src-cogencast]: [[source-cogencast]]
[^src-alphaflow]: [[source-alphaflow]]