---
title: "Algorithm Unrolling"
type: concept
tags:
  - optimization
  - deep-learning
  - interpretability
  - algorithm-design
  - network-architecture
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Algorithm Unrolling

**Algorithm Unrolling**（算法展开，也称 Deep Unfolding）是一种将迭代优化算法转换为神经网络架构的技术，通过将优化算法的每一步展开为网络的一层来实现[^src-cbsa]。

## 核心思想

传统方法中，优化算法（如梯度下降、ISTA）与神经网络是分离的：
- 优化算法：通过迭代求解目标函数
- 神经网络：通过反向传播学习参数

**Algorithm Unrolling** 的核心洞见是：许多迭代优化算法可以**展开**为前馈神经网络，其中：
- 迭代次数 → 网络层数
- 每步的更新规则 → 网络层的操作
- 步长、阈值等超参数 → 可学习的网络参数

这使得网络架构具有**可解释性**：每一层都对应优化目标的一个梯度/近端步骤[^src-cbsa]。

## 数学框架

### 优化问题

考虑一个正则化优化问题：

$$\min_x f(x) + \lambda g(x)$$

其中 $f$ 是光滑损失，$g$ 是非光滑正则项。

### ISTA 算法

迭代阈值收缩算法（ISTA）的每步：

$$x_{t+1} = \text{prox}_{\lambda}(x_t - \eta_t \nabla f(x_t))$$

其中 $\text{prox}_{\lambda}$ 是近端算子，$\eta_t$ 是步长[^src-cbsa]。

### Unrolled Network

将 T 步 ISTA 展开为 T 层网络：

```
Layer 1: x_1 = prox_λ(x_0 - η_0 ∇f(x_0))
Layer 2: x_2 = prox_λ(x_1 - η_1 ∇f(x_1))
...
Layer T: x_T = prox_λ(x_{T-1} - η_{T-1} ∇f(x_{T-1}))
```

- 近端算子 → 可学习的非线性层
- 步长 $\eta_t$ → 可学习的标量/向量参数
- 可以端到端训练[^src-cbsa]

## 在 CBSA 中的应用

Wen 等人使用算法展开从优化目标导出 CBSA：

### 1. 优化目标

从 MCR² 目标出发，引入代表性 token 概念后的压缩目标：

$$\min_Z R(Z) - \sum_{k=1}^{K} R(U_k^\top Q) - \lambda\|Z\|_0$$

### 2. 梯度步骤

对压缩项执行**一个**梯度下降步：

$$Z \leftarrow Z - \kappa \nabla_Z R_c(Q | U^{[K]})$$

这个梯度步骤直接对应 CBSA 的前向传播——**算法的一步 = 网络的一层**[^src-cbsa]。

### 3. CBSA 的展开结构

CBSA 包含两个阶段，对应优化目标的不同组成部分：

1. **代表性 token 提取**：对应约束满足（通过 cross-attention 近似 coding rate）
2. **代表性 token 收缩 + 广播**：对应压缩项的梯度下降步

这使得 CBSA 的每个操作都有明确的优化解释：不是在设计"好"的架构，而是在**展开优化目标的求解过程**[^src-cbsa]。

## Algorithm Unrolling 的优势

### 1. 可解释性

- 每层操作对应优化目标的明确数学含义
- 不需要"为什么这样设计"的启发式解释
- 可分析每层对目标的贡献[^src-cbsa]

### 2. 可学习超参数

- 步长 η、κ 等从数据学习，而非手动调参
- 适应不同数据分布和任务[^src-cbsa]

### 3. 理论基础

- 继承优化算法的收敛保证
- 可利用优化理论的工具分析网络行为[^src-cbsa]

## 其他应用场景

### 1. CSPN (Convolutional Sparse Coding by Neural Networks)

将卷积稀疏编码的迭代求解展开为网络，用于图像恢复[^src-cbsa]。

### 2. LISTA (Learned ISTA)

学习 ISTA 的阈值函数和步长，用于稀疏信号恢复[^src-cbsa]。

### 3. Deep Iterative Verification Networks

将迭代验证算法展开用于目标检测[^src-cbsa]。

### 4. CRATE / White-Box Transformer

CRATE 将 MCR² 目标的 ISTA 步骤展开为完整 Transformer：
- MSSA（Multi-head Subspace Self-Attention）：对应 MCR² 的压缩项梯度
- ISTA 模块：对应稀疏性惩罚和展开项的处理[^src-cbsa]

## 与 End-to-End 学习的对比

| 方面 | Algorithm Unrolling | End-to-End 训练 |
|------|---------------------|------------------|
| 可解释性 | 高（每层对应优化步骤） | 低（黑盒） |
| 参数效率 | 高（继承优化结构） | 需要更多数据 |
| 理论基础 | 强（优化理论） | 弱（经验） |
| 灵活性 | 中（受优化算法形式限制） | 高（任意架构） |
| 收敛保证 | 有（基于优化理论） | 无 |

## 引用

[^src-cbsa]: [[source-cbsa]]