---
title: "Building Normalizing Flows with Stochastic Interpolants"
type: source-summary
tags:
  - stochastic-interpolant
  - interflow
  - continuous-normalizing-flows
  - flow-matching
  - optimal-transport
  - generative-model
  - iclr-2023
created: 2026-07-13
last_updated: 2026-07-23
source_count: 1
confidence: high
status: active
---

# Building Normalizing Flows with Stochastic Interpolants

**Building Normalizing Flows with Stochastic Interpolants**（Albergo & Vanden-Eijnden, ICLR 2023；arXiv:2209.15571）提出用**随机插值**（stochastic interpolant）构造任意基分布 $\rho_0$ 与目标 $\rho_1$ 之间的连续时间归一化流，并给出速度场的二次、可仿真无关目标；所得生成模型称为 [[interflow|InterFlow]][^src-stochasticinterpolants]。

## 核心贡献

1. **随机插值过程**：给定满足 $I_{t=0}(x_0,x_1)=x_0$、$I_{t=1}(x_0,x_1)=x_1$ 的可微插值 $I_t$，独立采样 $x_0\sim\rho_0$、$x_1\sim\rho_1$ 定义 $x_t=I_t(x_0,x_1)$。默认三角插值 $I_t=\cos(\frac12\pi t)x_0+\sin(\frac12\pi t)x_1$[^src-stochasticinterpolants]。
2. **二次速度目标（Proposition 1）**：插值密度 $\rho_t$ 满足连续性方程，其速度 $v_t$ 是
   $$
   G(\hat v)=\mathbb{E}\big[|\hat v_t(I_t)|^2-2\,\partial_t I_t\cdot\hat v_t(I_t)\big]
   $$
   的唯一极小点；极小值为 $- \mathbb{E}[|v_t(I_t)|^2]$，且可用 $\tilde G(\hat v)=G(\hat v)+\mathbb{E}[|\hat v_t(I_t)|^2]\to 0$ 作收敛诊断[^src-stochasticinterpolants]。
3. **InterFlow 生成模型**：用参数化 $\hat v$ 最小化 $G$ 后，解概率流 ODE $\dot X_t=v_t(X_t)$ 实现双向采样与任意时刻似然；**无需对 ODE 求解器反传**，训练为仿真无关二次回归[^src-stochasticinterpolants]。
4. **可优化传输（Proposition 2）**：在插值上对 $\min_{\hat v}G$ 做最大化，在可插值密度假设下恢复 [[benamou-brenier-algorithm|Benamou–Brenier]] 动态最优传输；也可参数化基密度 $\rho_0$ 缩短路径[^src-stochasticinterpolants]。
5. **$W_2$ 控制（Proposition 3）**：近似速度诱导的终点密度与真目标的 $W_2^2$ 被 $H(\hat v)=\int_0^1\int|\hat v_t-v_t|^2\rho_t\,dx\,dt$ 与 Lipschitz 常数指数上界控制[^src-stochasticinterpolants]。
6. **与 score 扩散的对偶（Proposition 4）**：当 $\rho_0=\mathcal N(0,I)$ 且用三角插值时，$\nabla\log\rho_t$ 可由 $v_t$ 显式写出；速度在 $t\in[0,1]$ 有界，而对应 SDE 的漂移/扩散在端点奇异——论证可绕过扩散、直接用 ODE[^src-stochasticinterpolants]。

## 与同期方法

论文明确将自身与 [[source-rectified-flow|Rectified Flow]]（Liu et al.）及 [[source-flow-matching|Flow Matching]]（Lipman et al.）并列为**同期**仿真无关速度匹配工作；相对 score-SDE / Schrödinger bridge 强调有限时间、任意端点密度、直接概率流 ODE 与二次损失[^src-stochasticinterpolants]。对 rectification 的批评：迭代拉直要求每步映射精确，否则偏差累积[^src-stochasticinterpolants]。

## 实验要点

- **2D**：多模态/棋盘格与任意 $\rho_0\leftrightarrow\rho_1$ 数据集间插值；模式不粘连[^src-stochasticinterpolants]。
- **表格**（POWER/GAS/HEPMASS/MINIBOONE/BSDS300）：相对 FFJORD 等连续流 NLL 持平或更优（BSDS300 略逊），且相对 MLE-ODE 训练显著加速（MiniBooNE 约 400× 每 epoch）[^src-stochasticinterpolants]。
- **图像**：CIFAR-10 NLL 2.99 / FID 10.27；ImageNet $32\times32$ NLL 3.45 / FID 8.49；可扩展到 Oxford Flowers $128\times128$（此前 MLE CNF 难以达到）[^src-stochasticinterpolants]。

## 局限

- 默认插值一般**非**最优传输路径，需额外 max-min / 参数化 $a_t,b_t$ 或 $\rho_0$ 才能逼近 OT[^src-stochasticinterpolants]。
- 图像结果未用 EMA、截断等扩散工程技巧，FID 落后最强扩散基线[^src-stochasticinterpolants]。
- 完整 OT 插值学习留作未来工作；梯度场约束可消去无散度速度分量但不保证路径最优[^src-stochasticinterpolants]。

## 链接

- [[stochastic-interpolant]] — 随机插值概念
- [[interflow]] — InterFlow 生成模型
- [[flow-matching]] — 同期条件流匹配
- [[rectified-flow]] — 同期直线流 / reflow
- [[continuous-normalizing-flow]] — CNF 背景
- [[optimal-transport]] — 最优传输与 Benamou–Brenier
- [[building-schrodinger-bridges]] — SB 教程中的 stochastic interpolant 构造

[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
