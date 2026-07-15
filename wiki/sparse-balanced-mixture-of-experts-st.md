---
title: "Sparse Balanced Mixture of Experts for ST Graphs"
type: technique
tags:
  - mixture-of-experts
  - spatiotemporal-forecasting
  - graph-learning
  - load-balancing
  - sparse-activation
created: 2026-07-19
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# Sparse Balanced Mixture of Experts for ST Graphs

MAGE 提出的**稀疏平衡混合专家系统**（Sparse yet Balanced MoE）用于克服线性自适应图学习的低秩瓶颈[^src-mage]。

## 设计目标

线性核近似使 Rank(A) ≤ dG ≪ N，节点表示受限于低秩子空间。引入 K 个专家独立生成自适应图，可将表示矩阵秩提升至 ≤ min{d, K·dG}[^src-mage]：

$$H = \sum_{k=1}^{K} \alpha_k A^{(k)} H$$

当 K ≥ ⌈d/dG⌉ 时达到满秩 d[^src-mage]。

## 稀疏机制

从 KG 个候选专家（默认 16）中，每节点只激活 Top-K（默认 4）最相关的[^src-mage]：

$$\tilde{\alpha}_{ik} = \text{Sigmoid}(H_i^{(c-1)^\top} \theta_k + \gamma_k)$$

其中 γk ∈ R 是可学标量，趋向 +∞ 时强制激活，趋向 −∞ 时强制抑制。Sigmoid 使模型能产生尖锐的专家偏好[^src-mage]。

## 平衡机制

引入优先级调制器 βk，对历史使用频率过高的专家施加惩罚，使用不足则加分[^src-mage]：

$$\alpha_{ik} = \begin{cases} \tilde{\alpha}_{ik} + \beta_k, & k \in \arg\text{Top-K}\{\tilde{\alpha}_{ir} + \beta_r\} \\ 0, & \text{otherwise} \end{cases}$$

βk 通过符号 SGD 优化，目标为 βk 逼近实际激活次数与平均期望的差[^src-mage]：

$$\beta_k \leftarrow \beta_k - \mu \cdot \text{sgn}\left(N_k - \frac{N \cdot K}{K_G}\right)$$

其中 μ 是学习率（默认 10⁻³）。平衡机制确保所有专家被均等利用，每个专家约 6.25%(1/16) 激活率，避免某些专家"荒废"而模型坍缩到少数专家[^src-mage]。

## 每专家差分图

每个专家 k 分配 4 个可学嵌入 E₁(k)...E₄(k) ∈ RN×dG，生成差分图[^src-mage]：

$$A^{(k)} = \text{Softmax}(E_1^{(k)})\text{Softmax}(E_2^{(k)\top}) - \lambda \cdot \text{Softmax}(E_3^{(k)})\text{Softmax}(E_4^{(k)\top})$$

λ 通过 ω + exp(⟨λ₁,λ₂⟩) − exp(⟨λ₃,λ₄⟩) 重参数化保证数值稳定，ω∈(0,1) 为超参数[^src-mage]。

## 对比

与标准 Transformer MoE（如 Mixtral 的 FFN 级 MoE）不同，MAGE 的 MoE 作用于**图结构生成**层面——每个专家对应一种独特的空间拓扑假设，多专家混合实现了空间依赖的多样性建模[^src-mage]。

参见：[[mixture-of-experts]] — MoE 的通用框架和时空预测中的应用；[[low-dimensional-graph-adjacency]] — 低秩瓶颈概念及 GSNet 的解决方案。

[^src-mage]: [[source-mage]]
