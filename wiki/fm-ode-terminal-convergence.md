---
title: "FM ODE Terminal Convergence"
type: concept
tags:
  - flow-matching
  - ode-dynamics
  - convergence
  - manifold-hypothesis
  - theory
  - icml-2025
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# FM ODE 终端收敛

FM 模型在分布层面的收敛（$p_{t}$ 推前到 $p_1=p$）不保证单条 ODE 轨迹收敛到数据分布：轨迹可能绕着数据流形打转（winding）而不落入支撑[^src-2412-18730]。轨迹收敛（等价于流映射 $\Psi_1$ 在 $t=1$ 处存在）对生成稳定性和直接学习 $\Psi_1$ 的 [[consistency-models|Consistency Models]] 一类方法是前提条件[^src-2412-18730]。Wan、Wang、Mishne、Wang 的 ICML 2025 论文在温和假设下证明了 FM ODE 的终端收敛，作者称这是首个覆盖数据支撑在低维子流形上情形的结果[^src-2412-18730]。

## 两类终端奇异性

直接对 $\sigma$ 积分到 0 有障碍，论文辨析了两类奇异性（Appendix C.2）[^src-2412-18730]：

1. **公式奇异**：$t$ 参数下向量场分母 $\beta_t \to 0$，$\sigma$ 参数下有 $1/\sigma$ 因子；换用 $\lambda = \tfrac12\log\sigma^{-2}$ 坐标（$dx_\lambda/d\lambda = m_\lambda - x_\lambda$）看似消除分母，但收敛仍要求 $\|m_\lambda - x_\lambda\| \to 0$ 足够快——这正是需要证明的内容。
2. **几何奇异**：数据支撑非满支撑时，medial axis $\Sigma_\Omega$（投影不唯一的点集）处 denoiser 极限不连续、Lipschitz 常数爆炸。二点分布例子（$p=\tfrac12\delta_{-1}+\tfrac12\delta_1$）显式给出 $m_\sigma$ 的闭式：$m_\sigma \to f$（$x>0$ 取 1、$x<0$ 取 $-1$、$x=0$ 取 0）在 $\Sigma_\Omega=\{0\}$ 处不连续，且 $dm_\sigma/dx$ 在该点发散（Example C.5）。Proposition C.4 进一步给出：当 $p$ 非满支撑时，对几乎所有 $x \notin \Omega \cup \Sigma_\Omega$，$\lim_{t\to1}\|u_t(x)\| = \infty$。

Picard–Lindelöf 类标准 ODE 工具因此失效，收敛证明需要绕开这些奇点。

## Denoiser 收敛到投影

出发点是 denoiser 的终端行为（Theorem 5.1）：对有限 2-moment 的 $p$，一切 $x \in \mathbb{R}^d \setminus \Sigma_\Omega$ 满足

$$
\lim_{\sigma\to 0} m_\sigma(x) = \mathrm{proj}_\Omega(x),
$$

且带速率控制 $O(\sigma^\zeta)$（任意 $0<\zeta<1$；离散分布指数收敛、$m$ 维子流形上为 $m_\sigma + O(\sigma^2)$，Appendix D）。由于 $x$ 处投影为 $x$ 当且仅当 $x\in\Omega$，只要轨迹被吸到 $\Omega$ 附近，$\|m_\sigma(x_\sigma)-x_\sigma\|$ 就按需要消失[^src-2412-18730]。

## 主定理：流映射在 $t=1$ 存在

假设（Assumption 5.2）只要求：支撑 $\Omega$ 的 reach $\tau_\Omega > 0$（排除尖角与瓶颈，对球、环面等常见紧子流形成立），加上局部密度下界 $p(B_r(x)) \ge C_R r^k$（离散分布 $k=0$、$m$ 维流形 $k=m$）[^src-2412-18730]。则（Theorem 5.3）：

1. $\Psi_1(x) := \lim_{t\to1}\Psi_t(x)$ 对 $\mathbb{R}^d$ 中几乎处处的 $x$ 存在；
2. $\Psi_1$ 可测且 $(\Psi_1)_\# p_{\mathrm{prior}} = p$；
3. 收敛率 $\|\Psi_1(x) - \Psi_t(x)\| = O(\sigma_t^{\zeta/2})$[^src-2412-18730]。

细化（Theorem 5.4）：支撑在正 reach、有界第二基本形式的 $m$ 维闭子流形上时收敛率为 $O(\sqrt{\sigma_t})$；离散分布为 $O(\sigma_t)$。子空间情形的闭式解（Example 5.5，$\alpha_t=t,\beta_t=1-t$）给出 $\|x_1-x_t\| = \Theta(1-t) = \Theta(\sigma_t)$，说明离散情形速率最优，并支持流形情形或可从 $O(\sqrt{\sigma_t})$ 改进到 $O(\sigma_t)$ 的猜想（作者观点，依据是分布路径 $d_{W,2}(q_\sigma,p)=O(\sigma)$，Proposition 5.6）[^src-2412-18730]。

**实用推论**：终端阶段轨迹位移是 $O(\sigma_t^{\zeta/2})$ 量级的小量，少步采样在终端不牺牲质量[^src-2412-18730]。

## 与先前工作的关系

Pidstrigach (2022) 与 Permenter & Yuan (2024) 只证明轨迹靠近数据支撑时 denoiser 近似投影（"近似投影"解释）；Gao & Li (2024) 对离散测度分析局部簇吸收，但证明隐含假设 ODE 已收敛、且需要有界先验支撑（不适用于常见 FM 设定）；本文给出对几乎处处的点的一般收敛结果与全程轨迹刻画[^src-2412-18730]。同步工作 Baptista et al. (2025) 对经验分布的记忆动力学做了相似分析（同样用到 Voronoi 图）并提出正则化手段[^src-2412-18730]。

## 流映射的等变性

Theorem 5.3 建立了 $\Psi_1$ 后，可讨论其对几何变换的行为（Proposition 5.7）：对相似变换 $T(x)=\gamma(Ox+b)$（缩放+正交+平移），变换后分布的流映射与原流映射满足

$$
\Psi_t(Ox) = s_t(O\Psi_t(x) + \alpha_t b), \qquad \Psi_1(Ox) = \gamma(O\Psi_1(x) + b).
$$

推论（Remark 5.8）：数据支撑在任意仿射子空间 $A$ 时，可先投影到 $\mathbb{R}^{\dim A}$ 训练 FM，再经该等变关系扩展回环境空间——子空间情形的显式降维（Example 5.5：正交坐标不参与动力学，$x_t = (x_t^{\mathrm{sub}}, \beta_t y_0)$）由此推广到仿射子空间[^src-2412-18730]。这一等变性也是数据增强下稳定性的理论描述[^src-2412-18730]。

## 相关页面

- [[fm-ode-trajectory-stages]] — 初始/中间阶段的均值吸引与局部簇吸收
- [[diffusion-memorization-geometry]] — 离散支撑下的终端吸收与记忆现象
- [[consistency-models]] — 学习 $\Psi_1$ 的一步生成方法，以轨迹收敛为前提
- [[meanflow]] — 平均速度场一步生成框架
- [[one-step-flow-generation]] — 区间平均速度 + JVP 的一步生成技术
- [[probability-flow-ode]] — 概率流 ODE 与 FM ODE 的等价口径
- [[x-prediction]] — denoiser 的参数化视角

[^src-2412-18730]: [[source-2412-18730]]
