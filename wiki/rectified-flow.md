---
title: "Rectified Flow"
type: technique
tags:
  - generative-model
  - flow-matching
  - ode
  - few-step-generation
created: 2026-05-31
last_updated: 2026-06-08
source_count: 2
confidence: high
status: active
---

# Rectified Flow

**Rectified Flow** 是一种基于 ODE 的生成模型，通过迭代 **rectification**（校正）过程学习从源分布到目标分布的直线轨迹，实现 1-2 步的快速采样 [^src-rectified-flow]。

## 核心思想

传统扩散模型和流匹配模型学习弯曲的概率流 ODE 轨迹，需要多步数值积分才能生成高质量样本。Rectified Flow 通过 rectification 过程逐步"拉直"这些轨迹，使得少量欧拉步长即可逼近真实 ODE 解 [^src-rectified-flow]。

### Rectification 过程

给定源分布 $\pi_0$ 和目标分布 $\pi_1$，定义初始耦合 $(X_0, X_1)$，其中 $X_0 \sim \pi_0$，$X_1 \sim \pi_1$。Rectification 迭代执行以下步骤 [^src-rectified-flow]：

1. **从当前耦合采样**：采样配对数据 $(X_0^k, X_1^k)$，其中 $X_0^k \sim \pi_0$，$X_1^k$ 由当前 ODE 流从 $X_0^k$ 演化得到
2. **学习确定性映射**：训练神经网络 $v_\theta$ 最小化损失 $\mathbb{E}[\| v_\theta(X_0^k) - (X_1^k - X_0^k) \|^2]$
3. **定义新 ODE**：$\frac{dZ_t}{dt} = v_\theta(Z_t)$，$Z_0 = X_0$，此时 $Z_1$ 的分布更接近 $\pi_1$
4. **更新耦合**：新的耦合为 $(X_0, Z_1)$，返回步骤 1

理论保证：每次 rectification 缩短或保持轨迹长度，有限次后轨迹变为直线（无交叉）[^src-rectified-flow]。

## 关键特性

### 少步生成

- **1 步生成**：直接用 $X_1 = X_0 + v_\theta(X_0)$ 作为生成结果，质量较低但速度快
- **2 步生成**：$X_{0.5} = X_0 + 0.5 v_\theta(X_0)$，$X_1 = X_{0.5} + 0.5 v_\theta(X_{0.5})$，在 CIFAR-10 上达到 FID ≈ 4.85 [^src-rectified-flow]
- **4 步生成**：FID ≈ 4.23，接近 1000 步扩散模型的质量 [^src-rectified-flow]

### 与最优传输 (OT) 的关系

当初始耦合为独立耦合（$X_0 \perp X_1$）时，rectified flow 收敛到从 $\pi_0$ 到 $\pi_1$ 的最优传输映射（在凸代价下）[^src-rectified-flow]。这使得 Rectified Flow 与 [[optimal-transport|最优传输]] 理论紧密相连。

### Reflow：无监督变体

Reflow 是 Rectification 的无监督版本，仅使用从 $\pi_0$ 采样的数据，无需配对 $(X_0, X_1)$。通过最小化 $\mathbb{E}[\| v_\theta(X_0) - (\hat{X}_1 - X_0) \|^2]$ 来学习，其中 $\hat{X}_1$ 由当前 ODE 生成 [^src-rectified-flow]。

## 与相关技术的比较

| 方法 | 路径类型 | 采样步数 | 训练复杂度 |
|------|---------|---------|-----------|
| **Rectified Flow** | 直线 ODE | 1-4 步 | 适中 |
| [[flow-matching|Flow Matching]] | 预定义 ODE | 10-50 步 | 低 |
| [[diffusion-model|扩散模型]] | 随机 SDE | 50-1000 步 | 高 |
| [[consistency-models|Consistency Models]] | 直连 ODE | 1 步 | 高 |
| [[shortcut-models|Shortcut Models]] | 自洽 ODE | 1-2 步 | 高 |

## 理论保证

1. **单调性**：每次 rectification 的路径长度 $\ell_{k+1} \le \ell_k$ [^src-rectified-flow]
2. **收敛性**：$\lim_{k \to \infty} \ell_k = \ell_{\min}$，且路径变为直线（无交叉）[^src-rectified-flow]
3. **OT 收敛**：当初始耦合为独立耦合时，rectified flow 收敛到 $W_2$-OT 映射（凸代价）[^src-rectified-flow]

## 应用案例

### UrbanDiT 中的 InstaFlow

[[urbandit|UrbanDiT]]（Yuan et al., NeurIPS 2025）使用 Rectified Flow 训练方法 **InstaFlow** 实现 25× 加速 [^src-rectified-flow]：
- 将 DDPM 的 1000 步采样压缩到 40 步
- 保持生成质量的同时大幅降低推理延迟
- 成为 UrbanDiT 作为开放世界时空基础模型的关键技术支柱

### Shortcut Models

Rectification 思想启发了 [[shortcut-models|Shortcut Models]]（arXiv 2025），后者将自一致性推广到任意流模型，实现 1 步生成 [^src-rectified-flow]。

### FlowTS：第一个时间序列 Rectified Flow 模型

[[flowts|FlowTS]] (arXiv 2025) 是首个将 rectified flow 应用于时间序列生成的工作[^src-flowts]。与图像域需要 reflow 迭代不同，FlowTS 直接在时间序列域学习直线 ODE 轨迹，30 步采样即 SOTA（Context-FID Stocks 0.019 vs Diffusion-TS 0.067）[^src-flowts]。

## 局限性

- **初始耦合敏感**：独立耦合 vs OT 耦合的性能差异显著 [^src-rectified-flow]
- **高维数据挑战**：理论保证虽好，但在高维图像上实际效果受维数诅咒影响 [^src-rectified-flow]
- **Reflow 稳定性**：无监督 Reflow 需要仔细调参，可能不稳定

[^src-rectified-flow]: [[source-rectified-flow]]
[^src-flowts]: [[source-flowts]]
