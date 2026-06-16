---
title: "Gaussian Schrödinger Bridge"
type: concept
tags:
  - schrödinger-bridge
  - optimal-transport
  - gaussian-processes
  - bures-wasserstein
  - closed-form-solution
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: active
---

# Gaussian Schrödinger Bridge

当两个端点分布均为 Gaussian 时，Gaussian Schrödinger bridge 有**闭式解**[^src-schrodinger-bridges-generative-modeling]。给定 $\pi_0 = \mathcal{N}(\mu_0,\Sigma_0)$, $\pi_T = \mathcal{N}(\mu_T,\Sigma_T)$ 以及线性 SDE 驱动的 reference process：

$$\mathbb{Q}: \quad dX_t = (c_t X_t + \alpha_t)dt + \sigma_t dB_t$$

最优 SB path measure $\mathbb{P}^\star$ 本身是 Gaussian-Markov 的，均值和协方差均可解析计算。

## Static Gaussian Entropic Optimal Transport

The optimal coupling $\pi_{0,T}^\star$ is a joint Gaussian distribution:

$$\pi_{0,T}^\star = \mathcal{N}\left(\begin{bmatrix}\mu_0\\\mu_T\end{bmatrix}, \begin{bmatrix}\Sigma_0 & C_\sigma \\ C_\sigma^\top & \Sigma_T\end{bmatrix}\right)$$

with cross-covariance given by:

$$C_\sigma = \frac{1}{2}\left(\Sigma_0^{1/2} D_\sigma \Sigma_0^{1/2} - \sigma^2 I\right)$$

where $D_\sigma$ is the matrix square root:

$$D_\sigma = (4\Sigma_0^{1/2}\Sigma_T\Sigma_0^{1/2} + \sigma^4 I)^{1/2}$$

The parameter $\sigma$ (or $\sigma_t$ for time-varying diffusion) controls the entropic regularization strength. As $\sigma \to 0$, $C_\sigma$ approaches the classical Bures-Wasserstein optimal transport coupling.

## Dynamic Gaussian SB

The optimal path measure $\mathbb{P}^\star$ is Gaussian-Markov. Its marginals at time $t$ are:

$$p_t^\star = \mathcal{N}(\mu_t^\star, \Sigma_t^\star)$$

and the process follows a closed-form linear SDE:

$$dX_t = f_N(X_t,t)dt + \sigma_t dB_t$$

where $f_N$ is the optimal linear drift (explicitly computable from the mean and covariance trajectories).

### Mean Trajectory

$$\mu_t^\star = \bar{r}_t \mu_0 + r_t \mu_T + \zeta(t) - r_t\zeta(T)$$

where the time-weighting functions are:

$$\begin{aligned}
r_t &= \frac{\kappa(t,T)}{\kappa(T,T)} \\
\bar{r}_t &= \tau_t - r_t\tau_T \\
\tau_t &= \exp\left(\int_0^t c_s ds\right) \\
\sigma_\star^2 &= \frac{\kappa(T,T)}{\tau_T}
\end{aligned}$$

and $\kappa(s,t)$ is the Green's function for the reference SDE. The term $\zeta(t) = \int_0^t \exp(\int_s^t c_u du)\alpha_s ds$ accounts for the inhomogeneous drift $\alpha_t$.

### Covariance Trajectory

$$\Sigma_t^\star = \bar{r}_t^2 \Sigma_0 + r_t^2 \Sigma_T + r_t\bar{r}_t(C_{\sigma_\star} + C_{\sigma_\star}^\top) + \kappa(t,t)(1-\rho_t)I$$

where $\rho_t$ is a scalar function derived from the Green's function and $C_{\sigma_\star}$ is the cross-covariance evaluated at the effective regularization level $\sigma_\star$. The first two terms capture marginals at $t=0,T$, the third encodes the OT coupling structure, and the fourth adds entropic diffusion.

## Bures-Wasserstein Formulation

The covariance evolution can be understood variationally through the **Bures-Wasserstein metric** on the space of positive definite matrices. The optimal trajectory $\Sigma_t$ solves:

$$\inf_{\Sigma_t} \int_0^T \left[\frac{1}{2}\|\dot{\Sigma}_t\|_{\Sigma_t}^2 + U_\sigma(\Sigma_t)\right]dt$$

### Bures-Wasserstein Kinetic Energy

The squared metric is:

$$\|\dot{\Sigma}_t\|_{\Sigma_t}^2 = \frac{1}{2}\text{Tr}(\tilde{S}_t^\top \Sigma_t^{-1} \tilde{S}_t)$$

where $\tilde{S}_t$ is the symmetrized velocity $\tilde{S}_t = \dot{\Sigma}_t \Sigma_t^{1/2} \Sigma_t^{-1/2}$ (or equivalently the solution to $\Sigma_t \tilde{S}_t + \tilde{S}_t \Sigma_t = \dot{\Sigma}_t$). This metric captures the Riemannian geometry of transporting covariance matrices.

### Diffusion Potential

$$U_\sigma(\Sigma_t) = \frac{\sigma_t^4}{8}\text{Tr}(\Sigma_t^{-1})$$

This potential penalizes small eigenvalues of $\Sigma_t$, reflecting that entropy regularization (diffusion) resists collapsing the distribution into lower-dimensional subspaces. As $\sigma_t \to 0$, the potential vanishes and we recover the classical Bures-Wasserstein geodesic.

## Connection to General SB

| | Gaussian SB | General SB |
|---|-------------|------------|
| State space | $\mathbb{R}^d$ with Gaussian endpoints | Arbitrary $\mathbb{R}^d$ distributions |
| Solution | Closed-form in $\mu_t, \Sigma_t$ | Requires numerical optimization |
| Coupling | Gaussian $\pi_{0,T}^\star$ with $C_\sigma$ | General $\pi_{0,T}^\star = e^{\varphi+\hat{\varphi}-c}\pi_0\pi_T$ |
| Control | Linear drift $f_N$ | Nonlinear $u^\star = \sigma_t\nabla\log\varphi_t$ |

The Gaussian case serves as both a tractable analytical model and a building block for approximation methods (e.g., Gaussian approximations to non-Gaussian SB via moment matching).

## Relation to [[optimal-transport]]

In the limit $\sigma \to 0$, the Gaussian SB converges to the classical Gaussian optimal transport (Bures-Wasserstein) problem. The cross-covariance $C_\sigma$ reduces to the Bures-Wasserstein transport map covariance. Non-zero $\sigma$ adds entropic diffusion along the trajectory, smoothing the transport path.

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]