---
title: "Mixture of Experts (MoE)"
type: concept
tags:
  - machine-learning
  - architecture
  - scalability
created: 2026-04-29
last_updated: 2026-07-25
source_count: 7
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

## 在时间序列基础模型中的应用

[[moirai-moe|Moirai-MoE]] (ICML 2025) 是首个将 **Sparse MoE** 引入时间序列基础模型预训练的框架[^src-moirai-moe]：

- **架构**：在 6/12 层 decoder-only Transformer 中，每层 FFN 替换为 M=32 experts 的 MoE 层，每个 token 仅激活 K=2 experts
- **簇基门控**：提出 [[cluster-based-gating|新型门控函数]]——用预训练 dense 模型的 token 嵌入 k-means 聚类中心引导 expert 分配，在所有专家数量配置下一致优于随机初始化线性门控[^src-moirai-moe]
- **与标准 Sparse MoE 的差异**：Moirai-MoE 的 MoE 作用于时间序列 token 维度，通过 [[token-level-specialization|token 级专业化]] 替代频率级人工数据分组——浅层 expert 分配因频率而异、深层趋于一致，实现频率不变表示[^src-moirai-moe]
- **效果**：Moirai-MoE-S (11M activated, 117M total) 以 17% Monash 聚合 MAE 提升超越 dense Moirai-S，以 65× 更少激活参数优于 Chronos-L[^src-moirai-moe]
- **效率**：推理时间 273s vs Chronos-S 551s（受益于 patch size 16 vs Chronos 的 point-wise tokenization）[^src-moirai-moe]

同期工作 Time-MoE 也引入 MoE，但使用逐点 tokenization 和标准线性门控，Moirai-MoE 在零样本上显著优于 Time-MoE-B 和 Time-MoE-L[^src-moirai-moe]。

## 在动力系统重建中的应用

[[dynamix|DynaMix]] (NeurIPS 2025) 是首个将 MoE 用于动力系统重建（DSR）零样本泛化的模型[^src-dynamix]：

- **架构**：J=10 个 AL-RNN 专家 + 门控网络（CNN 上下文编码器 + 状态注意力机制 + MLP）
- **Dense MoE**：所有 10 个专家均被激活，通过 softmax 加权输出——与 FaST 的 Dense MoE 一致
- **状态注意力门控**：基于投影潜在状态与上下文观测之间的距离计算注意力权重 $w_t^{att}$，再经 CNN 特征加权后由 MLP 生成专家权重 $w_t^{exp}$
- **动力学专业化**：不同专家自然专业化到不同的动力学模式（如混沌区域的不同部分），专家使用权重可构建动力学相似度度量
- **关键发现**：J ≥ 5 个专家即足够，更多收益递减——实际 J=10 已达到性能平台期[^src-dynamix]

## 相关工作
- [[mage|MAGE]] (NeurIPS 2025) 在**图结构生成层面**引入 sparse-balanced MoE——每个专家对应一种独特的空间拓扑假设（自适应图），每节点 Top-K 稀疏激活 + 符号 SGD 负载均衡 βk。与标准 MoE（如 Mixtral 的 FFN 级路由）不同，MAGE 的 MoE 作用于图学习而非特征变换，用差分图 A(k)=Softmax(E₁)Softmax(E₂ᵀ)−λ Softmax(E₃)Softmax(E₄ᵀ) 增强多样性[^src-mage]。参见 [[sparse-balanced-mixture-of-experts-st]] 了解其稀疏平衡机制的技术细节。

- [[stamimputer|STAMImputer]] (arXiv 2025) 将 MoE 上移到框架外层，用观测专家 (O-Expert) 依据稀疏度特征动态路由时间/空间注意力专家，号称首次把 MoE 用于交通数据填补[^src-stamimputer]。

## 在多模态时序预测中的 Non-Fusion Guidance

[[timi|TiMi]] (ICML 2026) 提出 **[[mmoe|Multimodal Mixture-of-Experts (MMoE)]]**——一种将 MoE 用于跨模态知识引导而非特征融合的创新范式[^src-timi]。MMoE 包含两个互补的 MoE 子系统：**TMoE** 用 LLM 提取的文本因果知识作为门控信号路由 experts；**SMoE** 用全局序列表示路由 experts 捕获趋势[^src-timi]。与标准 Sparse/Dense MoE（仅基于输入 token 的 routing）不同，MMoE 的核心创新在于**门控信号的来源**——text-based routing 使 LLM 推理可以动态选择 experts 而非简单地将特征融合[^src-timi]。该模块可替换任意 Transformer backbone 的标准 FFN 层，PatchTST+MMoE 平均 MSE 降低 18.2%[^src-timi]。

## 相关技术

- [[glu-gated-linear-unit|Gated Linear Units (GLU)]]
- [[heterogeneous-moe-routing|异质性感知 MoE 路由]]
- [[adaptive-graph-agent-attention|自适应图代理注意力]]
- [[multi-modality-guided-spatial-expert|多模态引导空间专家]]
- [[token-level-specialization|Token 级专业化]]
- [[cluster-based-gating|簇基门控]]
- [[mmoe|MMoE (TiMi)]]
- [[timi|TiMi]]
- [[moirai-moe|Moirai-MoE]]

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-most]: [[source-most]]
[^src-dynamix]: [[source-dynamix]]
[^src-stamimputer]: [[source-stamimputer]]
[^src-mage]: [[source-mage]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-timi]: [[source-timi]]