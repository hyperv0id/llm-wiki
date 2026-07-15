---
title: "Building Schrödinger Bridges"
type: technique
tags:
  - schrödinger-bridge
  - generative-modeling
  - bridge-construction
created: 2026-06-16
last_updated: 2026-07-18
source_count: 3
status: active
---

# Building Schrödinger Bridges

六种互补的数学构造方法（此外 [[gmf|GMF]] 使用单步 Rectified Flow 作为 DSB 近似，无需 iterative proportional fitting 即可估计传输代价用于多模态融合的可靠性评估[^src-gmf]）。以下六种方法均收敛于同一 optimal Markov control drift $u^\star = \sigma_t\nabla\log\varphi_t$，该 drift 最小地修正 reference dynamics 以匹配 prescribed marginal distributions[^src-schrodinger-bridges-generative-modeling]。

## 1. Mixture of Conditional Bridges

将 dynamic SB 分解为 **static SB**（用于 endpoint coupling）加 conditional bridges：

$$\mathbb{P}^\star = \int \mathbb{Q}(\cdot|x_0,x_T)\,\pi_{0,T}^\star(dx_0,dx_T)$$

其中 $\pi_{0,T}^\star$ 求解 [[entropic-optimal-transport|static SB]]，$\mathbb{Q}(\cdot|x_0,x_T)$ 是 conditioned on endpoints 的 reference bridge。

**核心洞察**：optimal Schrödinger bridge 是 endpoint-conditioned reference bridges 的**概率加权混合**。

## 2. Time Reversal

给定 forward SDE $dX_t = f dt + \sigma_t dB_t$，reverse-time process $X_s = X_{T-s}$ 满足：

$$d\tilde{X}_s = [-f + \sigma_{T-s}^2\nabla\log p_{T-s}]ds + \sigma_{T-s} d\tilde{B}_s$$

**score correction** $\nabla\log p_{T-s}$ 补偿 probability flow。这是 score-based generative modeling 的基础。

## 3. Forward-Backward SDEs (FBSDEs)

将 state $X_t$ 与 log-potentials $Y_t = \log\varphi_t(X_t)$ 和 $\hat{Y}_t = \log\hat{\varphi}_t(X_t)$ 耦合：

$$\begin{aligned}
dX_t &= (f + \sigma_t Z_t)dt + \sigma_t dB_t \quad &(Z_t = \nabla\log\varphi_t) \\
dY_t &= \frac{1}{2}\|Z_t\|^2 dt + Z_t^\top dB_t \\
d\hat{Y}_t &= [\nabla\cdot(\sigma_t\hat{Z}_t - f) + \frac{1}{2}\|\hat{Z}_t\|^2 + \sigma_t^2 Z_t^\top\hat{Z}_t]dt + \hat{Z}_t^\top dB_t \quad &(\hat{Z}_t = \nabla\log\hat{\varphi}_t)
\end{aligned}$$

terminal constraint: $Y_T + \hat{Y}_T = \log\pi_T(X_T)$。

## 4. Doob's h-Transform

通过 $h(x,t) = \mathbb{E}_Q[\varphi_T(X_T)|X_t=x]$ tilting reference process：

$$\mathbb{P}^h = \frac{h(X_T, T)}{h(X_0, 0)}\mathbb{Q}$$

变换后的过程满足 $dX_t = (f + \sigma_t^2\nabla\log h)dt + \sigma_t dB_t$，恢复 $u^\star = \sigma_t\nabla\log h = \sigma_t\nabla\log\varphi_t$。

## 5. Markovian and Reciprocal Projections

**Markovian projection** $\mathrm{proj}_{\mathcal{M}}(\Pi)$：最接近 bridge mixture 的 Markov measure。

**Reciprocal projection** $\mathrm{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{M})$：强制 endpoint constraints 同时保持 bridge 结构。

交替投影（Iterative Markovian Fitting, IMF）收敛到 $\mathbb{P}^\star$：

$$\mathbb{P}^\star = \lim_{n\to\infty} (\mathrm{proj}_{\mathcal{R}(\mathbb{Q})} \circ \mathrm{proj}_{\mathcal{M}})^n(\Pi_0)$$

### Pythagorean Identity

$$\forall \mathbb{P} \in \mathcal{R}(\mathbb{Q}), \mathbb{M} \in \mathcal{M}: \mathrm{KL}(\mathbb{P}\|\mathbb{M}) = \mathrm{KL}(\mathbb{P}\|\mathrm{proj}_{\mathcal{M}}(\mathbb{P})) + \mathrm{KL}(\mathrm{proj}_{\mathcal{M}}(\mathbb{P})\|\mathbb{M})$$

该恒等式保证 IMF 中 KL 单调递减。

## 6. Stochastic Interpolants

将 bridge 表示为 $x_t = I^\star(x_0, x_T, t) + \gamma(t)z$，其中 $z \sim \mathcal{N}(0,I_d)$，$\gamma(0)=\gamma(T)=0$，$I^\star$ 求解：

$$\max_{\hat{I}}\min_{\hat{u}} \mathbb{E}\!\left[\frac{1}{2}\|\hat{u}\|^2 - (\partial_t\hat{I} + \dot{\gamma}z - \epsilon\gamma^{-1}z)\cdot\hat{u}\right]$$

边界条件：$I(x_0,x_T,0)=x_0$，$I(x_0,x_T,T)=x_T$。

原论文 [[source-stochasticinterpolants|Building Normalizing Flows with Stochastic Interpolants]] 给出无额外噪声桥时的二次目标 $G(\hat v)$、InterFlow ODE 生成，以及与 score 扩散的对偶；详见 [[stochastic-interpolant]] 与 [[interflow]][^src-stochasticinterpolants]。

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
[^src-gmf]: [[source-gmf]]
