---
title: "Linear Adaptive Graph Learning"
type: technique
tags:
  - graph-learning
  - spatiotemporal-forecasting
  - kernel-method
  - linear-complexity
  - scalability
created: 2026-07-19
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# Linear Adaptive Graph Learning

**线性自适应图学习**（Linear Adaptive Graph Learning）是 MAGE 提出的将自适应图卷积复杂度从 O(N²dG) 降至 O(N·d·dG) 的 kernel 近似技术[^src-mage]。

## 动机

传统自适应图卷积需要显式计算 N×N 相似度矩阵 S=E₁E₂ᵀ，复杂度 O(N²dG)，无法扩展到大路网[^src-mage]。BigST 使用正随机特征（PRF）近似，GSNet 使用低秩压缩，但都造成表达能力损失[^src-mage]。

## 方法

从去 ReLU 的自适应图 A=Softmax(E₁E₂ᵀ) 出发。节点 vi 的图卷积输出为[^src-mage]：

$$H_i^{(c)} = \frac{\sum_j \text{Sim}(E_{1i}, E_{2j}) H_j^{(c-1)}}{\sum_m \text{Sim}(E_{1i}, E_{2m})}$$

其中 Sim(·,·) 是正定核。关键洞察：将非负激活前置于内积之前，用 Φ:E₁↦exp(E₁+η), Ψ:E₂↦exp(E₂+ξ) 保证内积自然非负[^src-mage]：

$$\text{Sim}(e_i^{(1)}, e_j^{(2)}) = \langle\Phi(e_i^{(1)}), \Psi(e_j^{(2)})\rangle$$

利用乘法结合律，先算子表达式的右侧部分（Ψ(e₂) 与 H 的内积），再与左侧 Φ(e₁) 内积，避免显式构建 N×N 矩阵[^src-mage]：

$$H^{(c)} = \text{Softmax}(E_1) \cdot \text{Softmax}(E_2^\top) \cdot H^{(c-1)}$$

计算顺序为 Softmax(E₂ᵀ)·H(c-1) → Softmax(E₁)·(结果)，复杂度 O(2·N·d·dG)，线性于 N[^src-mage]。

## 低秩瓶颈

代价是 [[low-dimensional-graph-adjacency|Rank(A)]] = Rank(Softmax(E₁)Softmax(E₂ᵀ)) ≤ min{N, dG} = dG ≪ N[^src-mage]。节点表示被限制在低维子空间中。MAGE 通过多专家策略克服此瓶颈（见 [[sparse-balanced-mixture-of-experts-st]]）[^src-mage]。

[^src-mage]: [[source-mage]]
