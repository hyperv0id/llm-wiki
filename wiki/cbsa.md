---
title: "Contract-and-Broadcast Self-Attention (CBSA)"
type: entity
tags:
  - attention
  - interpretable-attention
  - efficient-attention
  - algorithm-unrolling
  - mcr2
  - linear-attention
  - neurips-2025
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Contract-and-Broadcast Self-Attention (CBSA)

**CBSA**（Contract-and-Broadcast Self-Attention）是 Wen 等人（NeurIPS 2025）提出的一种通过算法展开（algorithm unrolling）从优化目标推导出的注意力机制[^src-cbsa]。

## 核心思想

CBSA 的核心思想是**压缩所有 token 通过收缩少数代表性 token**：将所有输入 token 压缩到低维子空间，同时通过少数代表性 token（representatives）来捕捉输入数据的几何和信息论本质[^src-cbsa]。

## 数学定义

### 优化目标

CBSA 源自修改后的 MCR² 目标，引入代表性 token $Q = q(Z) \in \mathbb{R}^{d \times m}$（$m \ll N$ 为代表性 token 数量）：

$$\max_Z R(Z) - \sum_{k=1}^{K} R(U_k^\top Q) - \lambda\|Z\|_0 \quad \text{s.t.} \quad |R(U_k^\top Q) - R(U_k^\top Z)| \leq \tau, \forall k \in [K]$$

其中 $R(\cdot)$ 是 coding rate（编码率），$U_k^{[K]} = \{U_k\}_{k=1}^K$ 是 K 个 p 维子空间的正交基[^src-cbsa]。

### CBSA 算子

通过对压缩项执行梯度下降步骤，导出 CBSA 算子：

$$\text{CBSA}(Z | U^{[K]}) = Z - \kappa \nabla_Z R_c(Q | U^{[K]})$$

具体形式为（简化版，使用 softmax 近似）：

$$\text{CBSA}(Z | U^{[K]}) \approx \sum_{k=1}^{K} U_k^\top \underbrace{U_k^\top Q \cdot \text{softmax}((U_k^\top Q)^\top (U_k^\top Q))}_{\text{收缩 via 自注意力}} \underbrace{A_k^\top}_{\text{广播}}$$

其中：
- **Contraction（收缩）**：代表性 token 之间的自注意力操作，等效于梯度方向
- **Broadcast（广播）**：复用提取阶段的注意力矩阵 $A_k$ 将收缩广播回所有输入 token[^src-cbsa]

## CBSA 作为统一注意力公式

CBSA 通过改变代表性 token 的选择，可以实例化为多种不同的注意力机制：

| 注意力机制 | 代表性 token 的选择 | 特点 |
|------------|---------------------|------|
| **Softmax Attention (MSSA)** | 自表达代表：$Q = Z, m = N$ | 输入 token 本身作为代表，压缩在 N 维空间进行[^src-cbsa] |
| **Linear Attention** | 正交代表：通过对 $U_k^\top Z$ 执行 SVD 得到主方向 | 沿主方向动态压缩，因式分解注意力矩阵实现线性复杂度[^src-cbsa] |
| **Channel Attention (TSSA)** | 固定正交代表：$L_k = I_p$ | 沿固定轴压缩，特征通道按二阶矩自适应缩放[^src-cbsa] |
| **Agent Attention** | 无收缩步骤 | 移除 contraction 步骤，等效于 Agent Attention[^src-cbsa] |

## 计算复杂度

| 机制 | 复杂度 |
|------|--------|
| MSSA (Softmax Attention) | $O(2Nd^2 + 2N^2d)$ |
| **CBSA** | **$O(2Nd^2 + 3Nmd + 2m^2d)$** |
| Linear Attention | $O(Nd^2)$ |

当固定 $m = p = d/K$ 且 $N > 2d/K$ 时，CBSA 的 FLOPs 低于 MSSA，达到**线性复杂度**[^src-cbsa]。

## 与现有注意力机制的关系

### 与 CRATE/MSSA 的关系

- MSSA 通过 MCR² 目标的梯度步骤导出，但存在 Gram 矩阵导致 $O(N^2)$ 复杂度[^src-cbsa]
- CBSA 引入代表性 token 概念，用 $m \ll N$ 个代表替代全部 N 个 token 进行压缩计算，绕过 Gram 矩阵瓶颈[^src-cbsa]

### 与 Linear Attention 的关系

- CBSA 的 linear attention 变体与标准 linear attention 高度相似（等效于 factorized attention matrix）[^src-cbsa]
- CBSA 的理论解释更清晰：通过压缩目标驱动，而非启发式矩阵分解[^src-cbsa]

### 与 QUEST 的关系

| 方面 | CBSA | QUEST |
|------|------|-------|
| 目标 | 可解释性 + 效率（通过压缩目标） | 训练稳定性 + 鲁棒性（通过键归一化） |
| 方法 | 算法展开（优化目标 → 网络架构） | 键 ℓ2 归一化 |
| 复杂度 | 线性 O(N)（固定 m） | 二次 O(N²) |
| 解释性 | 白盒（梯度步骤可解释） | 黑盒（归一化效果事后解释） |

## 实验结果

### ImageNet-1K 分类

| 模型 | 参数量 | FLOPs | Top-1 |
|------|--------|-------|-------|
| ViT-S | 22.1M | 9.8G | 72.4 |
| **CBT-S (Ours)** | **6.7M** | **4.0G** | **71.4** |
| CRATE-B | 22.8M | 12.6G | 70.8 |

CBT-S 以 ViT-S 30% 的参数量和 40% 的 FLOPs 达到可比精度[^src-cbsa]。

### 鲁棒性

- 参数扰动下 CBSA 极为鲁棒（子空间基的扰动不显著改变其张成的子空间）[^src-cbsa]
- 对抗攻击和数据损坏下性能下降幅度小于标准注意力

### 语义分割（ADE20K）

CBT decoder 以 Segmenter 20% 的 FLOPs 实现 +1.5% mIoU 提升[^src-cbsa]。

## 局限性

1. **子空间假设**：union of linear subspaces 可能不适用于所有模态（论文排除 NLP ��务）[^src-cbsa]
2. **预训练适配**：从预训练 ViT 适配时性能略低于 linear attention[^src-cbsa]
3. **早期层解压缩**：早期层出现"解压缩"现象的原因不明[^src-cbsa]

## 引用

[^src-cbsa]: [[source-cbsa]]