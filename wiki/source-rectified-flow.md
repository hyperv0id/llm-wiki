---
title: "Rectified Flow: Flow Straight and Fast"
type: source-summary
tags:
  - generative-model
  - flow-matching
  - ode
  - optimal-transport
created: 2026-05-31
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Rectified Flow: Flow Straight and Fast

**Rectified Flow** 是一种基于 ODE 的生成模型，通过学习从源分布到目标分布的直线轨迹，实现快速少步生成 [^src-rectified-flow]。

## 核心贡献

1. **Straightening ODE Paths**：提出 Rectification 过程，通过迭代学习更直的概率流 ODE 轨迹，消除流线中的曲率和交叉 [^src-rectified-flow]。
   - 初始路径可以任意（例如，独立耦合、最优传输耦合）
   - 每次 rectification 从当前 ODE 采样配对数据 `(X₀, X₁)`，重新学习确定性映射
   - 理论保证：rectification 不会增加路径长度，经过有限次 rectification 后轨迹变为直线 [^src-rectified-flow]

2. **Few-Step Generation**：与依赖扩散模型的多步去噪不同，Rectified Flow 在训练后 1-2 步即可生成高质量样本 [^src-rectified-flow]。
   - 单次 rectification 通常足够，无需迭代多次
   - 在 CIFAR-10 上 2 步生成达到 FID ≈ 4.85，4 步 ≈ 4.23
   - 在 LSUN 卧室数据集上 2 步生成语义合理图像 [^src-rectified-flow]

3. **Reflow 耦合**：Rectified Flow 可以与 **Reflow** 耦合实现更优的传输路径 [^src-rectified-flow]。
   - Reflow 是 Rectification 的无监督变体，仅使用从源分布采样的数据，无需配对数据
   - 结合 Reflow 和 OT 初始化可进一步提升 1-2 步生成质量

4. **理论保证**：
   - Rectification 严格缩短或保持流路径长度
   - 有限次 rectification 后轨迹变为直线（无交叉）[^src-rectified-flow]
   - 当初始耦合为独立耦合时，rectified flow 等价于从源分布到目标分布的最优传输映射（在凸代价下）[^src-rectified-flow]

## 与 Flow Matching 的关系

Rectified Flow 与 [[flow-matching]] 及 [[stochastic-interpolant|Stochastic Interpolants]] 密切相关但目标不同 [^src-rectified-flow][^src-stochasticinterpolants]：
- **Flow Matching**：训练连续归一化流 (CNF) 以匹配目标向量场，需要预定义的概率路径（通常是高斯路径）
- **Rectified Flow**：通过 rectification 直接学习直线轨迹，天然具有少步生成能力，无需预定义路径
- **Stochastic Interpolants / InterFlow**：固定（可优化）插值后用二次目标学概率流速度，强调任意端点密度与 max-min 通往动态 OT[^src-stochasticinterpolants]
- **互补性**：Flow Matching 可通过 Rectification 进一步优化，反之亦然

## 实验评估

- **CIFAR-10**：2 步 FID 4.85（vs DDPM 的 1000 步），4 步 FID 4.23 [^src-rectified-flow]
- **LSUN 卧室**：2 步生成语义合理图像，质量优于 1000 步 DDPM [^src-rectified-flow]
- **ImageNet 32×32**：单次 rectification 后 2 步生成 FID 7.61 [^src-rectified-flow]

## 影响与应用

Rectified Flow 启发了后续许多少步生成的工作：
- **InstaFlow**（UrbanDiT 使用）：基于 Rectified Flow 的训练方法，实现 25× 加速 [^src-rectified-flow]
- **Shortcut Models**：将 rectification 推广到更少步的通用框架 [^src-rectified-flow]
- 在 ODE/SDE 求解器设计中，rectification 成为减少数值误差的核心技术

## 局限性

- 初始耦合的选择影响最终质量（独立耦合 vs OT 耦合）[^src-rectified-flow]
- Reflow 变体需要仔细调参，可能在无监督场景下不稳定
- 理论保证虽然优雅，但在高维图像数据上实际效果受维数诅咒影响 [^src-rectified-flow]

[^src-rectified-flow]: [[source-rectified-flow]]
[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
