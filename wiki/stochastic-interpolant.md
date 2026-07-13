---
title: "Stochastic Interpolant"
type: concept
tags:
  - stochastic-interpolant
  - continuous-normalizing-flows
  - flow-matching
  - generative-model
  - optimal-transport
  - probability-flow-ode
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Stochastic Interpolant

**Stochastic interpolant（随机插值）** 是 Albergo & Vanden-Eijnden（ICLR 2023）提出的、在有限时间 $[0,1]$ 上连接任意基密度 $\rho_0$ 与目标密度 $\rho_1$ 的随机过程构造：独立采样端点后经可微插值映射 $I_t$ 得到 $x_t$，其边缘密度 $\rho_t$ 满足连续性方程，对应速度场由**二次目标**唯一刻画[^src-stochasticinterpolants]。

## 定义

取 $I_t:\mathbb{R}^d\times\mathbb{R}^d\to\mathbb{R}^d$，满足边界条件
$$
I_{t=0}(x_0,x_1)=x_0,\qquad I_{t=1}(x_0,x_1)=x_1,
$$
并在温和可积条件下 $\mathbb{E}[|\partial_t I_t|^2]<\infty$。定义
$$
x_t = I_t(x_0,x_1),\qquad x_0\sim\rho_0,\; x_1\sim\rho_1\text{ 独立。}
$$
$\{x_t\}$ 称为 stochastic interpolant；$\rho_t=\mathrm{Law}(x_t)$ 为**插值密度**[^src-stochasticinterpolants]。

常用**三角插值**（便于与 Gaussian base 的 score 对偶）：
$$
I_t(x_0,x_1)=\cos\!\big(\tfrac12\pi t\big)\,x_0+\sin\!\big(\tfrac12\pi t\big)\,x_1.
$$
更一般的线性族 $I_t=a_t x_0+b_t x_1$（$a_0=b_1=1,a_1=b_0=0$）可参数化并学习以缩短路径[^src-stochasticinterpolants]。

## 连续性方程与速度

形式上
$$
\rho_t(x)=\int \delta\big(x-I_t(x_0,x_1)\big)\,\rho_0(x_0)\rho_1(x_1)\,dx_0 dx_1,
$$
电流 $j_t$ 由 $\partial_t I_t$ 的同类期望给出。在 $\rho_t>0$ 处令 $v_t=j_t/\rho_t$，则
$$
\partial_t\rho_t+\nabla\cdot(v_t\rho_t)=0,\qquad \rho_{t=0}=\rho_0,\;\rho_{t=1}=\rho_1.
$$
概率流 ODE $\dot X_t=v_t(X_t)$ 将 $\rho_0$ push-forward 到 $\rho_t$，终点映射实现生成模型 [[interflow|InterFlow]][^src-stochasticinterpolants]。

## 二次变分刻画

$v$ 是目标
$$
G(\hat v)=\mathbb{E}\big[|\hat v_t(I_t)|^2-2\,\partial_t I_t\cdot\hat v_t(I_t)\big]
$$
的唯一极小点；极小值 $G(v)=-\mathbb{E}[|v_t(I_t)|^2]$ 等于负路径动能，故对 $I_t$（及可选的 $\rho_0$ 参数）最大化 $\min_{\hat v}G$ 可缩短传输、并在可插值密度假设下恢复 [[benamou-brenier-algorithm|Benamou–Brenier]] 最优传输[^src-stochasticinterpolants]。

与朴素回归 $\mathbb{E}[|\hat v_t(I_t)-\partial_t I_t|^2]$ 不同：后者在真 $v$ 处一般**不**归零（$|v|^2\le|\partial_t I|^2$ 通常严格），而 $G$ 提供可监控的诊断 $\tilde G\to 0$[^src-stochasticinterpolants]。

## 与相关范式

| 范式 | 端点 | 时间 | 训练 | 动力学 |
|------|------|------|------|--------|
| Stochastic interpolant / InterFlow | 任意 $\rho_0,\rho_1$ | 有限 $[0,1]$ | 二次、仿真无关 | 概率流 ODE |
| [[flow-matching\|Flow Matching]] | 通常 Gaussian→数据 | 有限 | CFM 条件回归 | ODE |
| [[rectified-flow\|Rectified Flow]] | 任意耦合 | 有限 | 直线速度 + reflow | ODE |
| Score-SDE / [[score-based-sde]] | 数据→噪声 | 常需 $T\to\infty$ | score 匹配 | SDE + 概率流 |
| [[schrodinger-bridge\|Schrödinger bridge]] | 规定两端边际 | 有限 | 熵正则路径 OT | SDE 控制 |

同期工作中，[[source-flow-matching|Flow Matching]] 与 [[source-rectified-flow|Rectified Flow]] 给出类似仿真无关速度学习；SI 强调**任意插值族**、**路径与目标解耦**、以及 max-min 通往动态 OT 的理论路线[^src-stochasticinterpolants]。在 [[building-schrodinger-bridges|SB 构造综述]] 中，随机插值亦作为第六种桥构造出现（可加噪声桥 $I+\gamma(t)z$）[^src-stochasticinterpolants]。

## Gaussian base 与 score

当 $\rho_0=\mathcal N(0,I)$ 且用三角插值时，速度与 score 可互推（Proposition 4），因而 SI 速度可导出扩散式采样；但速度本身在端点有界，而逆向 SDE 系数在 $t=0,1$ 奇异——凸显“直接 ODE”路径的数值优势[^src-stochasticinterpolants]。

## 推广

- **因子化插值**：按簇 $k$ 分解 $\rho_0,\rho_1$，仅在簇内插值，便于多模态 OT 近似[^src-stochasticinterpolants]。
- **梯度速度**：在梯度场上最小化 $G$ 得到 $\nabla\phi_t$，消去对密度演化无影响的无散度分量；与 KILBO 目标相关但避免 $x,t$ 导数[^src-stochasticinterpolants]。

## 链接

- [[source-stochasticinterpolants]] — 原始 ICLR 2023 论文摘要
- [[interflow]] — 基于 SI 的生成模型与实验
- [[flow-matching]] — 条件流匹配
- [[rectified-flow]] — 直线流与 reflow
- [[continuous-normalizing-flow]] — CNF / Neural ODE 背景
- [[optimal-transport]] — OT 与动态公式
- [[benamou-brenier-algorithm]] — 动态 OT 求解
- [[probability-flow-ode]] — 概率流 ODE
- [[building-schrodinger-bridges]] — SB 中的 SI 视角

[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
