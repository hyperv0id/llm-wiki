---
title: "CRATE (White-Box Transformer)"
type: entity
tags:
  - transformer
  - interpretable-attention
  - mcr2
  - mssa
  - white-box
  - subspace-learning
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# CRATE (White-Box Transformer)

**CRATE**（White-Box Transformer/Coding Rate reduction Transformer）是基于 MCR² 目标的 White-Box Transformer 架构，通过算法展开将 MCR² 优化目标的求解过程展开为完整的 Transformer[^src-cbsa]。

## 背景

### Black-Box vs White-Box

| 方面 | Black-Box (标准 ViT) | White-Box (CRATE) |
|------|---------------------|-------------------|
| 设计依据 | 经验/实验 | 优化目标 |
| 可解释性 | 低 | 高 |
| 每层操作 | Attention + MLP | 明确对应优化步骤 |
| 理论保证 | 无 | 有（优化理论） |

### MCR² 作为设计驱动

CRATE 的核心洞见：不从经验设计注意力机制，而是从明确的学习目标（MCR²）**推导**出注意力机制[^src-cbsa]。

## 架构组件

### 1. MSSA (Multi-head Subspace Self-Attention)

MSSA 是 CRATE 的 token mixer，通过 MCR² 目标的压缩项梯度步骤导出：

$$\text{MSSA}(Z | U^{[K]}) = \sum_{k=1}^{K} U_k^\top \underbrace{\text{softmax}(U_k^\top Z Z^\top U_k)}_{\text{Gram 矩阵}} U_k^\top Z$$

**关键**：
- 注意力矩阵 = Gram 矩阵 $U_k^\top Z Z^\top U_k$
- 来源：对压缩项 $R_c(Z | U^{[K]}) = \sum_k R(U_k^\top Z)$ 求梯度
- 对应 $O(N^2)$ 复杂度（来自 Gram 矩阵）[^src-cbsa]

### 2. ISTA 模块

CRATE 的前馈网络来自 ISTA（Iterative Shrinkage-Thresholding Algorithm）：

$$x_{t+1} = \text{prox}_{\lambda}(x_t - \eta_t \nabla f(x_t))$$

- 处理 MCR² 中的稀疏性惩罚 $\|Z\|_0$
- 对应展开后的前馈层[^src-cbsa]

### 3. 子空间基

每个注意力头建模为 K 个正交子空间：

$$U^{[K]} = \{U_k \in O(d, p)\}_{k=1}^K$$

- 可学习参数
- 对应 K 个子空间（可理解为 K 个"类别/簇"）[^src-cbsa]

## 与 CBSA 的关系

| 方面 | CRATE (MSSA) | CBSA |
|------|--------------|------|
| 目标 | MCR² | 修改后 MCR²（引入代表） |
| 注意力矩阵 | Gram 矩阵 $U_k^\top Z Z^\top U_k$ | 通过代表间接计算 |
| 复杂度 | O(N²)（全连接） | O(N)（固定 m） |
| 代表性 token | 无（所有 token 作为代表） | 有（m << N） |
| 可解释性 | 高 | 高 |
| 注意力变体 | 仅 softmax | softmax/linear/channel/agent |

### CBSA 是 CRATE 的高效变体

- CBSA 通过引入代表性 token 概念，绕过 Gram 矩阵计算
- 当 m = N 时，CBSA 退化为 MSSA
- CBSA 揭示了 MSSA 与其他注意力机制（linear、channel、agent）的统一[^src-cbsa]

## 实验性能

### ImageNet-1K 分类

| 模型 | 参数量 | FLOPs | Top-1 |
|------|--------|-------|-------|
| CRATE-B | 22.8M | 12.6G | 70.8 |
| CRATE-L | 77.6M | 43.3G | 71.3 |

相比 ViT-S（72.4），CRATE-B 参数更多但精度略低[^src-cbsa]。

### 可解释性验证

1. **紧凑结构化表示**：CRATE 学习到的表示确实是紧凑且结构化的（编码率分析验证）
2. **涌现分割特性**：仅用分类监督训练，CRATE 涌现出语义分割能力（无需分割监督）[^src-cbsa]

## CRATE + CBT 混合模型

论文发现将 CRATE（前半部 MSSA）+ CBT（后半部 CBSA）组合可获得最佳效果：
- 前半部 MSSA 涌现分割特性
- 后半部 CBSA 保持效率
- 分割特性不仅涌现，而且在后续层持续增强（而非衰减）[^src-cbsa]

## 引用

[^src-cbsa]: [[source-cbsa]]