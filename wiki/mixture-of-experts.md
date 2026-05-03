---
title: "Mixture of Experts (MoE)"
type: concept
tags:
  - machine-learning
  - architecture
  - scalability
created: 2026-04-29
last_updated: 2026-05-03
source_count: 2
references:
  - [[source-fast-long-horizon-forecasting]]
confidence: high
status: active
---

# Mixture of Experts (MoE)

## 定义

Mixture of Experts（MoE）是一种神经网络架构设计，通过多个专门的子网络（称为"experts"）来处理输入数据的不同部分，并使用一个**门控网络（gating network）**动态地将输入路由到最相关的 experts[^src-fast-long-horizon-forecasting]。

## 核心组件

### 1. Experts
- 多个独立的子网络，每个 expert 可以专门处理输入的不同方面
- 在 FaST 中，使用 **Gated Linear Units (GLU)** 作为 experts，相比传统 FFN 有更好的并行性

### 2. Gating Network
- 动态计算每个 expert 的权重
- 根据输入特征决定哪些 experts 被激活
- 关键挑战：**Expert 极化**（expert polarization）— 少量 experts 主导路由决策

## Dense MoE vs Sparse MoE

FaST 使用的是 **Dense MoE**，这与常见的 Sparse MoE（只激活 top-k experts）不同：

- **Dense MoE**: 所有 e 个 experts 都被激活，输出是所有 experts 的加权和
  - 计算：∑_{i=1}^{e} G_ℓ[:,i] ⊗ Exp_i(Z_t^ℓ)
  - FaST 通过 GLU 并行化实现高效计算
  
- **Sparse MoE**: 只激活 top-k experts（如 Switch Transformer），减少推理计算量但增加实现复杂度

FaST 选择 Dense MoE 的原因：GLU experts 的并行计算可以在单个 GPU 上高效完成，避免了跨设备分配 experts 的额外开销。

## 异质性感知路由（HA-Router）

在 FaST 中提出的 HA-Router 通过以下方式避免 expert 极化：

- **吸收原始时间序列模式**：使用原始输入 X_t 计算 expert 分数，而非仅使用当前层特征
- **注入自适应空间和时间 expert bias**：RS（空间位置）、R_t^T（日内时间）和 R_t^W（周内时间）

公式：
```
G_ℓ = softmax(g_ℓ(X_t) ⊕ R_S,ℓ ⊕ R_t^T,ℓ ⊕ R_t^W,ℓ) ∈ R^(N×e)
```

这使得节点能够选择与自身相关的 experts，而不是集中在单一 expert 上。

## 优势

- **并行计算效率**：GLU experts 可以在单个计算单元上并行处理，提升 1.4x 推理速度[^src-fast-long-horizon-forecasting]
- **异质性建模**：不同节点和时间段可以使用不同的 experts
- **避免 expert 极化**：HA-Router 通过异质性感知避免负载不平衡

## 在时空图预测中的应用

FaST 首次将 MoE 应用于大规模长视野时空图预测：
- 使用 HA-MoE 进行时间压缩输入
- 在 backbone 中用 HA-MoE 替换 FFN
- 实现 O(N·e·d) 空间复杂度（线性于节点数）

[[most|MoST]] (KDD 2026) 将 MoE 应用于空间依赖建模，提出 [[multi-modality-guided-spatial-expert|多模态引导空间专家]]：使用两类专家——模态共享专家（每种激活模态一个）和路由专家（由路由器基于模态融合嵌入选择）——来捕获区域特定的局部空间模式[^src-most]。每个专家通过交叉注意力建模传感器与其 top-k 最近邻的交互，而非全图关系[^src-most]。与 FaST 的 Dense MoE 不同，MoST 使用 Top-1 稀疏路由并引入负载均衡损失防止专家坍塌[^src-most]。

## 相关技术

- [[glu-gated-linear-unit|Gated Linear Units (GLU)]]
- [[heterogeneous-moe-routing|异质性感知 MoE 路由]]
- [[adaptive-graph-agent-attention|自适应图代理注意力]]
- [[multi-modality-guided-spatial-expert|多模态引导空间专家]]

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-most]: [[source-most]]