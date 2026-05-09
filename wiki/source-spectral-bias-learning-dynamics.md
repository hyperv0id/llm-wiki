---
title: "An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models"
type: source-summary
tags:
  - diffusion-models
  - spectral-bias
  - learning-dynamics
  - theory
  - kernel-regression
  - neurips-2025
  - harvard
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: high
status: active
---

# An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models

**Binxu Wang, Cengiz Pehlevan** (Kempner Institute, Harvard University)，NeurIPS 2025，arXiv:2503.03206v3。[^src-spectral-bias]

---

## 核心贡献

本文对扩散模型训练过程中生成分布的演化进行了**严格理论分析**，是首个对"扩散模型为什么先学低频"这一现象给出严密数学解答的工作。核心方法是利用**高斯等价原理**（Gaussian equivalence），将线性 denoiser 的全批量梯度流动力学求解为闭式解，进而积分概率流 ODE 得到生成分布的解析表达式。[^src-spectral-bias]

---

## 理论框架

### 问题设定

假设 denoiser 在每个噪声尺度 $\sigma$ 上是线性的（仿射）：

$$D(x; \sigma) = W_\sigma x + b_\sigma$$

各 $\sigma$ 的参数独立。训练使用 EDM 框架下的全批量 denoising score matching (DSM) 损失。[^src-spectral-bias]

### 高斯等价性

对于任意分布 $p_0$，线性 denoiser 的二次损失仅依赖于数据的一二阶矩 $(\mu, \Sigma)$。因此，任何数据分布的线性 denoiser 训练等价于在高斯近似 $N(\mu, \Sigma)$ 上的训练。[^src-spectral-bias]

### Diffusion = Ridge Regression

全批量 DSM 损失的线性 denoiser 等价于 ridge regression：

$$L_\sigma = \mathbb{E}_{x_0 \sim p_0} \|W x_0 + b - x_0\|^2 + \sigma^2 \|W\|_F^2$$

最优解为 $W_\sigma^* = \Sigma(\Sigma + \sigma^2 I)^{-1}$，即经典 ridge regression 解。[^src-spectral-bias]

---

## 核心定理：反比方差谱定律

### 一层线性 Denoiser (Prop. 4.2)

在协方差 $\Sigma$ 的特征基 $u_k$（对应特征值 $\lambda_k$）上，权重动力学退化为独立模式：

$$W_\sigma(\tau) = W_\sigma^* + \sum_{k=1}^d \left(W_\sigma(0) - W_\sigma^*\right) u_k u_k^\top e^{-2\eta\tau(\sigma^2 + \lambda_k)}$$

生成分布的方差沿各模式的演化为：

$$\tilde{\lambda}_k(\tau) = \sigma_T^2 \frac{\Phi_k^2(\sigma_0, \tau)}{\Phi_k^2(\sigma_T, \tau)}$$

其中 $\Phi_k$ 包含指数积分函数 $Ei$。**关键结论**：模式首次到达目标方差的时间满足：

$$\tau_k^* \propto \lambda_k^{-\alpha},\quad \alpha \approx 1$$

即**方差为 1/10 的模式学习时间约长 10 倍**。[^src-spectral-bias]

### 两层对称线性网络 (Prop. 5.1)

深度增加不改变谱偏置的本质。两层对称网络 $D(x, \sigma) = P_\sigma P_\sigma^\top x$ 中，出现时间同样满足 $\tau_k^* \propto \lambda_k^{-1}$，仅权重动力学从指数变为 sigmoidal。[^src-spectral-bias]

### 全宽度循环卷积 (Prop. 5.3)

卷积核与信号等大 $(K=N)$ 时，傅里叶域动力学与全连接层等价，但**权重共享带来 $N$ 倍加速**（$N$ 为信号维度），不改变反比方差定律。[^src-spectral-bias]

### 局部卷积 / Patch 卷积 (Prop. 5.4)

卷积核小于信号 $(K < N)$ 时，滤波器学习动力学退化为 patch 空间上的 ridge regression：

$$w_\sigma(\tau) = w_\sigma^* + \exp\left(-2N\eta\tau(\sigma^2 I + \Sigma_{\text{patch}})\right)(w_\sigma(0) - w_\sigma^*)$$

权重共享带来 $N$ 倍加速，但**局部性使多个傅里叶模式耦合**，可能改变谱偏置。[^src-spectral-bias]

---

## 理论总结

| 架构 | 权重动力学 | 分布动力学 | 加速因子 |
|------|-----------|-----------|---------|
| 一层线性 | 指数 (PC) | sigmoidal, 幂律 | — |
| 两层对称 | sigmoidal (PC) | sigmoidal, 幂律 | — |
| 全宽卷积 | 指数 (傅里叶) | sigmoidal, 幂律 | $\times N$ |
| Patch卷积 | 指数 (patch PC) | N.S. | $\times N$ |

谱偏置在以上所有架构中均存在：高方差模式先收敛，低方差模式后收敛。[^src-spectral-bias]

---

## 实验验证

### MLP-UNet（FFHQ）

- 生成方差 $\tilde{\lambda}_k(\tau)$ 沿特征基呈 sigmoidal 轨迹
- 高方差模式先突破，低方差模式后突破
- 出现时间 $\tau^*$ 与目标方差 $\lambda_k$ 呈幂律关系：$\alpha_{\text{incr}} = -0.48$（R²=0.97），$\alpha_{\text{decr}} = -0.35$（R²=0.92）
- 高斯数据上指数更接近 1（$d=256$ 时 $\alpha = 1.08$）[^src-spectral-bias]

### CNN-UNet（FFHQ）

- **谱偏置几乎消失**（$\alpha \approx 0$）：所有模式几乎同时出现
- 原因：局部卷积使相邻像素耦合，多个傅里叶模式绑定为一个学习单元
- 窄网络（ch=4）仍呈现谱偏置，与 patch-convolution 理论一致
- 宽网络（ch=128）模式同时出现，接近实用 UNet 行为[^src-spectral-bias]

### 关键结论

1. MLP 架构中谱偏置定律稳健成立
2. 卷积架构设计（局部性 + 宽度）重塑了学习动力学
3. 网络宽度与输入通道之比可能决定理论预测的偏差程度

---

## 与现有工作的关系

- **F-Principle / Spectral Bias (Rahaman et al., 2019)**：经典谱偏置关注前向传播的 NTK / 函数逼近，本文关注生成分布的训练动力学
- **Hidden Linear Structure (Wang & Vastola, 2023)**：揭示了扩散模型 score 的隐式线性结构，本文在此基础求解学习动力学
- **Diffusion is Spectral Autoregression (Dieleman, 2024)** / **Wavelet Score-Based (Guth et al., 2022)**：记录采样阶段的谱偏置，本文研究训练阶段
- **Kamb & Ganguli (2024)**：分析卷积扩散模型的创造力（来自约束的创意性），本文关于 local convolution 的分析可由此视角理解
- **SAGD / frequency-diffusion**：SAGD 操控前向噪声频谱塑造偏置，本文解释"为什么会存在这个偏置"

---

## 局限性

- 假设线性 denoiser、全批量梯度流、正交初始化——现实训练中这些条件都被违反
- 但 MLP 实验显示理论预测仍定性正确（仅幂指数修正）
- CNN-UNet 的完整理论处理留待未来工作
- 未分析 attention、normalization 等现代架构组件

## 引用

[^src-spectral-bias]: [[source-spectral-bias-learning-dynamics]]