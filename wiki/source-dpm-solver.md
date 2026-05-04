---
title: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"
type: source-summary
tags:
  - diffusion-models
  - sampling
  - fast-inference
  - nips-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# DPM-Solver

**DPM-Solver** 是由 Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu 于 2022 年发表在 NeurIPS 的论文（arXiv:2206.00927），提出了基于扩散 ODE 半线性结构的专用快速求解器，可在约 10 步内生成高质量样本 [^src-dpm-solver]。

## 核心贡献

### 1. 扩散 ODE 的精确解公式

DPM-Solver 利用扩散 ODE 的**半线性结构**（线性项 $f(t)x_t$ + 非线性项神经网络），通过"常数变易公式"推导精确解：

$$
x_t = \frac{\alpha_t}{\alpha_s} x_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda} \hat{\varepsilon}_\theta(\hat{x}_\lambda, \lambda) d\lambda
$$

其中 $\lambda_t = \log(\alpha_t / \sigma_t)$ 是半对数信噪比（half-log-SNR）。这一公式通过变量替换将线性部分解析计算，避免了相应离散误差。

### 2. 高阶求解器

基于精确解公式和指数积分器理论，论文提出了一阶到三阶的 DPM-Solver：

| 求解器 | NFE/步 | 收敛阶 |
|--------|--------|--------|
| DPM-Solver-1 | 1 | $\mathcal{O}(h^1)$ |
| DPM-Solver-2 | 2 | $\mathcal{O}(h^2)$ |
| DPM-Solver-3 | 3 | $\mathcal{O}(h^3)$ |

### 3. DDIM 等价于 DPM-Solver-1

论文证明 DDIM 的一步更新与 DPM-Solver-1 完全相同，从而揭示了 DDIM 之所以优于传统 Euler 方法的原因：DDIM 充分利用了扩散 ODE 的半线性结构。

### 4. 训练免费

与知识蒸馏等方法不同，DPM-Solver 无需额外训练，可"即插即用"应用于任意预训练的 DPM。

## 实验结果

- **CIFAR-10**: 4.70 FID @ 10 NFE, 3.24 FID @ 15 NFE, 2.87 FID @ 20 NFE
- **对比 RK 方法**: 同阶条件下 DPM-Solver 显著优于传统 RK 方法
- **多数据集**: CelebA, ImageNet, LSUN 上均实现 4~16× 加速

## 与现有方法的关系

- **DDPM**: 需要 ~1000 步采样，DPM-Solver 仅需 ~10 步
- **DDIM**: 等价于 DPM-Solver-1
- **Score-Based SDE**: ODE 求解器的重大改进

## 引用

[^src-dpm-solver]: Cheng Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps". NeurIPS 2022. arXiv:2206.00927