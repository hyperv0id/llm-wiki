---
title: "Stochastic Optimal Control for Schrödinger Bridges"
type: technique
tags:
  - schrödinger-bridge
  - stochastic-optimal-control
  - hjb-equation
  - bellman-principle
  - feynman-kac
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: medium
status: superseded
superseded_by: [[stochastic-optimal-control-sb]]
---

# Stochastic Optimal Control for Schrödinger Bridges

Stochastic optimal control (SOC) reformulates the [[schrodinger-bridge|Schrödinger bridge]] problem as a controlled diffusion problem: find the drift adjustment that minimizes expected running cost plus a terminal cost, subject to the initial distribution constraint[^src-schrodinger-bridges-generative-modeling].

## SOC Formulation

The control objective is:

$$\inf_u \mathbb{E}_{\mathbb{P}^u}\left[\int_0^T \left(\frac{1}{2}\|u(X_t^u,t)\|^2 + c(X_t^u,t)\right)dt + \Phi(X_T^u)\right]$$

subject to the controlled SDE:

$$dX_t^u = (f(X_t^u,t) + \sigma_t\, u(X_t^u,t))\,dt + \sigma_t\, dB_t, \qquad X_0^u \sim \pi_0$$

The state-dependent running cost is $\frac{1}{2}\|u\|^2 + c$, where $\frac{1}{2}\|u\|^2$ penalizes control effort and $c(x,t)$ is a potential cost. $\Phi$ is the terminal cost.

## Value Function and Bellman's Principle

Define the **value function** (optimal cost-to-go):

$$V_t(x) = \inf_u J(x,t;u)$$

Bellman's Principle of Optimality gives the recursive characterization:

$$V_t(x) = \inf_u \mathbb{E}\left[\int_t^\tau \left(\frac{1}{2}\|u\|^2 + c\right)ds + V_\tau(X_\tau^u)\;\big|\;X_t^u=x\right]$$

Taking the limit $\tau \to t$ yields the HJB equation.

## HJB Equation

The value function satisfies the Hamilton-Jacobi-Bellman equation:

$$\partial_t V_t = -(A_t V_t) + \frac{\sigma_t^2}{2}\|\nabla V_t\|^2 - c$$

with terminal condition $V_T = \Phi$, where $A_t$ is the **uncontrolled generator**:

$$A_t V = \langle f, \nabla V\rangle + \frac{\sigma_t^2}{2}\Delta V$$

The term $\frac{\sigma_t^2}{2}\|\nabla V_t\|^2$ makes the HJB equation **nonlinear** — the core difficulty that the [[hopf-cole-transform|Hopf-Cole transform]] resolves.

## Optimal Control

The optimal control is the negative gradient of the value function:

$$u^\star = -\sigma_t \nabla V_t$$

This is the pointwise minimizer of the Hamiltonian $H(x,u,\nabla V) = \langle u, \nabla V\rangle + \frac{1}{2}\|u\|^2$.

## Hopf-Cole Link

Applying the Hopf-Cole transformation $\varphi_t = e^{-V_t}$ linearizes the HJB equation:

$$\partial_t \varphi + \langle f, \nabla \varphi\rangle + \frac{\sigma_t^2}{2}\Delta \varphi - c\varphi = 0$$

with terminal condition $\varphi_T = e^{-\Phi}$. When the running cost $c=0$, this reduces to the forward Kolmogorov equation for the reference process.

## Feynman-Kac Solution

The value function admits a closed-form Feynman-Kac representation:

$$V_t(x) = -\log \mathbb{E}_Q\left[\exp\left(-\int_t^T c(X_s,s)\,ds - \Phi(X_T)\right)\;\big|\;X_t=x\right]$$

where the expectation is taken under the reference measure $\mathbb{Q}$.

## Loss Functions for Learning

Three families of losses arise from the SOC formulation[^src-schrodinger-bridges-generative-modeling]:

### Relative Entropy (RE) Loss

$$\mathcal{L}_{\text{RE}}(u) = \mathbb{E}_{\mathbb{P}^u}\left[\frac{1}{2}\int_0^T \|u\|^2\,dt + \Phi(X_T^u) + \int_0^T c\,dt\right]$$

Directly derived from the [[schrodinger-bridge|SB]] KL objective via Girsanov's theorem. Unbiased estimator of $\mathrm{KL}(\mathbb{P}^u\|\mathbb{P}^\star)$.

### Cross Entropy (CE) Loss

$$\mathcal{L}_{\text{CE}}(u) = \mathbb{E}_{\mathbb{P}^\star}\left[\frac{1}{2}\int_0^T \|u\|^2\,dt + \int_0^T \langle u, u^\star\rangle\,dt\right]$$

Convex in $\mathbb{P}^u$, enabling stable gradient-based optimization when samples from $\mathbb{P}^\star$ are available.

### Variance and Log-Variance Losses

Additional loss families designed to reduce variance in Monte Carlo gradient estimation, particularly beneficial for high-dimensional problems.

> [!note] CE vs RE
> CE loss requires access to samples from the optimal bridge $\mathbb{P}^\star$ and is convex in the path measure, making it theoretically attractive. RE loss works with rollouts from $\mathbb{P}^u$ and is more practical during early training when $\mathbb{P}^\star$ is not yet known.

## SB-SOC Connection

The SB problem is recovered as a special case of SOC. The optimal control satisfies:

$$u^\star = \sigma_t \nabla \log \varphi_t = -\sigma_t \nabla V_t$$

so the value function is simply $V_t = -\log \varphi_t$, where $\varphi_t$ is the forward Schrödinger potential. The **terminal cost** that recovers the SB marginal constraint is:

$$\Phi(x) = \log\frac{\hat{\varphi}_T(x)}{\pi_T(x)}$$

This choice eliminates the initial value function bias: the SB formulation directly specifies both endpoint distributions rather than relying on a penalty.

## Related

- [[schrodinger-bridge]] — the path-space problem that SOC reformulates
- [[hopf-cole-transform]] — the linearization that resolves HJB nonlinearity
- [[doob-h-transform]] — alternative construction via probability tilting
- [[diffusion-schrodinger-bridge-matching]] — practical algorithm based on SOC objectives

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
