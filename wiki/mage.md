---
title: "MAGE"
type: entity
tags:
  - spatiotemporal-forecasting
  - graph-learning
  - mixture-of-experts
  - linear-complexity
  - neurips-2025
created: 2026-07-19
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# MAGE

**MAGE** (Mixture of Adaptive Graph Experts) 是 USTC 团队在 NeurIPS 2025 提出的高效时空图预测框架，以线性计算复杂度实现自适应图学习[^src-mage]。

## 核心设计

MAGE 解决传统自适应图学习（A=Softmax(ReLU(E₁E₂ᵀ))）的三大缺陷[^src-mage]：

### 1. 去 ReLU → 线性核近似

ReLU 存在**[[edge-noise-amplification|边噪声放大]]**问题：负边权被放大为正，正边权被抑制，导致噪声边混入自适应图[^src-mage]。移除 ReLU 后，引入 [[linear-adaptive-graph-learning|kernel-based 近似]]——将指数激活前置到内积之前（Φ/Ψ → R+），利用乘法结合律避免显式构建 N×N 相似度矩阵，复杂度从 O(N²dG) 降至 O(N·d·dG)[^src-mage]。

代价是低秩瓶颈：Rank(A) = Rank(Softmax(E₁)Softmax(E₂ᵀ)) ≤ dG ≪ N[^src-mage]。

### 2. [[sparse-balanced-mixture-of-experts-st|稀疏平衡 MoE]]

KG 个候选专家（默认 16），每节点 Top-K 稀疏激活（默认 4），通过可学偏置 γk + 负载均衡调制器 βk 实现[^src-mage]：

- **稀疏**：α̃ik = Sigmoid(H(c-1)_iᵀ θk + γk)，γk → +∞ 强制激活，γk → −∞ 强制抑制[^src-mage]
- **平衡**：βk 按历史使用频率调整，使用过度则减分，使用不足则加分，符号 SGD 优化 βk ← βk − μ·sgn(Nk − N·K/KG)[^src-mage]

### 3. 差分图学习

每个专家分配 4 个可学嵌入 E₁⋯E₄，最终图 = 正图 − λ·负图：A(k) = Softmax(E₁)Softmax(E₂ᵀ) − λ Softmax(E₃)Softmax(E₄ᵀ)[^src-mage]。λ 通过 ω + exp(⟨λ₁,λ₂⟩) − exp(⟨λ₃,λ₄⟩) 重参数化以保证数值稳定[^src-mage]。

## 整体架构

L 层 MAGE Block（默认 3 层），每层 = MAGE(H) + FFN(H) with SwiGLU + spatiotemporal position embedding P[^src-mage]。输入表示 Z(0) = X·W₀ + b₀ + P ∈ RN×d[^src-mage]。

## 关键结果

- 17 数据集 / 14 基线 / 94% 指标 SOTA[^src-mage]
- 比 D²STGNN 快 118–960×，比 PatchSTG 快 4.7×，省显存 1.72×[^src-mage]
- MAGE 单步图卷积等价于传统方法多步卷积（理论证明）[^src-mage]
- 纯线性配置（16:0 Linear:Full）已 Pareto-最优，加全秩图无额外收益[^src-mage]

## 关联

- [[source-gwnet|GWNet]] — ReLU 自适应图的来源，MAGE 揭示了其[[edge-noise-amplification|边噪声放大]]缺陷
- [[bigst]] — 同为线性复杂度，用 [[positive-random-features|PRF]] 近似；MAGE 用 kernel 近似 + MoE 全面超越
- [[graphsparsenet|GSNet]] — 同为线性复杂度，用低秩矩阵分解；MAGE 走 MoE 路线克服低秩瓶颈
- [[mixture-of-experts]] — MoE 通用框架，MAGE 是首个在图结构生成层面应用 sparse-balanced MoE 的工作
- [[low-dimensional-graph-adjacency]] — 低秩瓶颈的理论背景

参见子技术页：[[linear-adaptive-graph-learning]]、[[sparse-balanced-mixture-of-experts-st]]

[^src-mage]: [[source-mage]]
