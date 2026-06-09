---
title: "Micro-Macro Coupled Koopman Modeling"
type: concept
tags:
  - koopman-operator
  - multi-scale-modeling
  - traffic-prediction
  - history-free-prediction
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: medium
status: active
---

# Micro-Macro Coupled Koopman Modeling

A unified prediction framework where both microscopic vehicle dynamics and macroscopic traffic flow evolution are jointly modeled using time-invariant Koopman operators in a shared linear observation space, introduced by [[mmckm|MMCKM]] (ICLR 2026)[^src-mmckm].

## Koopman Operator Theory

The Koopman operator $\mathcal{K}$ provides a framework for analyzing nonlinear dynamical systems through linearization in an infinite-dimensional observation space. For a discrete-time system $x_{t+1} = f(x_t)$, the Koopman operator acts on observables $g$ as $\mathcal{K}g = g \circ f$. In the lifted observation $z_t = g(x_t)$, the evolution becomes linear: $z_{t+1} = \mathcal{K} z_t$. Finite-dimensional approximation via DMD (Dynamic Mode Decomposition) or neural networks enables practical applications[^src-mmckm].

**Critical advantage for prediction tasks**: The Markovian property — when observation functions are time-invariant, prediction uses only the current state information without requiring historical trajectories[^src-mmckm].

## MMCKM Architecture

### Macro Koopman Evolution

The traffic graph $G_t$ is lifted to an observation space $Z_t$ via a GNN encoder $\phi_Z$, then evolved by a learnable linear matrix $K_Z$[^src-mmckm]:

$$Z_t = \phi_Z(G_t), \quad Z_{t+1} = K_Z Z_t, \quad \rho_{t+1} = \psi_Z(Z_{t+1})$$

The encoder/decoder are trained with reconstruction losses: $\|\phi_Z(G_{t+1}) - K_Z\phi_Z(G_t)\|_2^2$ (encoder consistency) and $\|\bar{\rho}_{t+1} - \rho_{t+1}\|_1$ (density prediction)[^src-mmckm].

### Micro Koopman Evolution with Control

Vehicle states $x_t^e$ are lifted via MLP encoder $\phi_z$, evolved with Koopman control theory[^src-mmckm]:

$$z_t = \phi_z(x_t^e), \quad z_{t+1} = K_z z_t + B_z u_t, \quad p_{t+1}^e = \psi_z(z_{t+1})$$

where $u_t = \text{CA}(z_t, Z_t)$ is a CrossAttention block injecting macroscopic flow influence. The CrossAttention design — rather than a simple linear projection — captures the context-dependent nature of traffic influence: vehicles in different positions contribute unequally[^src-mmckm].

同样基于 Koopman 线性化的 [[k2vae|K²VAE]]（ICML 2025）把这一思想用于通用概率时间序列预测：用 MLP 测量函数 + one-step eDMD 拟合 Koopman 算子构造"有偏线性系统"，再用引入控制输入的 KalmanNet 精炼——与 MMCKM 的微观 Koopman 控制项 $z_{t+1}=K_zz_t+B_zu_t$ 共享"Koopman 线性系统 + 控制项"的通用模式[^src-k2vae]。

## Spectral Alignment

A spectral alignment loss $L_{\text{spec}}$ couples the Koopman operator $K$ with the graph-PDE operators[^src-mmckm]:

$$L_{\text{spec}} = \min_{\Pi} \left(\|\text{Re}(\lambda(\theta)) - \Pi\lambda(L^{\text{diff}})\|_2^2 + \|\text{Im}(\lambda(\theta)) + \Pi\omega(C^{\text{adv}})\|_2^2\right)$$

where $\theta = \frac{1}{\Delta t}\log(K)$ (principal matrix logarithm via numerically stable real Schur form with Tikhonov regularization on near-unit eigenvalues), $\Pi$ is a permutation, and $\lambda(\cdot)$ denotes eigenvalues[^src-mmckm]. This explicitly couples Koopman dynamics to the learned graph-PDE operators — diffusion eigenvalues govern Koopman magnitude (decay), advection eigenvalues govern Koopman rotation (oscillation frequency)[^src-mmckm].

## ISS Stability Guarantee

The Koopman control path satisfies Input-to-State Stability (ISS)[^src-mmckm]:

$$\exists c \geq 1, \lambda \in (0,1): \|z_t\| \leq c\lambda^t\|z_0\| + \frac{c B_z}{1-\lambda}\sup_{0 \leq \tau \leq t-1}\|u_\tau\|$$

Enforced by: (1) $\kappa(K_z) < 1$ via bounded spectral radius (parameterized as $R = \kappa_{\text{max}} \cdot \sigma(\eta)$ where each real/complex block eigenvalue modulus < 1); (2) Bounded $u_t$ via Sigmoid on CrossAttention output; (3) Constrained $B_z = B_{\text{max}} \cdot \tanh(B_z)$ to prevent unbounded actuation gains during training[^src-mmckm]. This guarantees errors decay geometrically with rate $\kappa(K_z)$ and external influences remain bounded — no unbounded error growth over iterative Koopman applications[^src-mmckm].

[^src-mmckm]: [[source-mmckm]]
[^src-k2vae]: [[source-k2vae]]
