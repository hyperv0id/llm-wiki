---
title: "Sparse Teacher Forcing"
type: technique
tags:
  - training
  - dynamical-systems
  - rnn
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: medium
status: active
---

# Sparse Teacher Forcing (STF)

**稀疏教师强制（STF）** 是专为动力系统重建（DSR）设计的控制论训练方法，由 Mikhaeil et al. (NeurIPS 2022) 发展理论，Brenner et al. (ICML 2022) 首次应用于 DSR[^src-dynamix].

## 动机

在混沌系统上训练 RNN 面临根本困境[^src-dynamix]：
- **标准教师强制**（每步用真实数据输入）→ 梯度稳定但模型无法"探索未来"，无法捕捉长期动力学
- **自由运行**（完全自回归前向迭代）→ 梯度爆炸，训练崩溃

STF 通过间歇性重校准解决了这一困境。

## 机制

按固定间隔 τ，将当前潜在状态 $z_t$ 替换为从数据通过（伪）逆解码器推断的状态 $\tilde{z}_t$：

$$z_{t+1} = \begin{cases} F_\theta(\tilde{z}_t, C) & \text{if } t \in \mathcal{T} = \{l\tau + 1\}_{l \in \mathbb{N}_0} \\ F_\theta(z_t, C) & \text{otherwise} \end{cases}$$

其中 $\tilde{z}_t = (x_t, z_{N+1:M,t})^T$，即用观测数据替换潜在状态的前 N 个读出维度[^src-dynamix]。

## 关键参数

- **τ（重校准间隔）**：理论上应基于系统的 Lyapunov 谱选择，实践中作为超参数
- **仅用于训练**：测试时关闭，模型完全自回归前向迭代
- DynaMix 中 τ=10，消融显示 τ 的选择对零样本 DSR 质量至关重要[^src-dynamix]

## 在 [[dynamix|DynaMix]] 中的重要性

消融实验中，将 STF 替换为标准 BPTT 训练导致零样本 DSR 性能急剧下降，是 DynaMix 成功的最关键组件之一[^src-dynamix]。这表明训练算法、架构选择和训练语料库共同构成了 DSR 基础模型的"DSR 包"。

[^src-dynamix]: [[source-dynamix]]
