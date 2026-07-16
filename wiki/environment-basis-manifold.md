---
title: "Environment Basis Manifold"
type: technique
tags:
  - spatio-temporal-forecasting
  - environment-modeling
  - sparse-mixture
  - heterogeneity
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# Environment Basis Manifold

Environment Basis Manifold 是 [[stpde|STPDE]] 框架中负责**参数化空间异质性与非平稳性**的组件。它与 [[invariant-diffusion-operator|Invariant Diffusion Operator]] 形成"普适规律–环境调制"的解耦设计[^src-stpde]。

## 结构

使用 $K$ 个共享的可学习基底 $\Phi = \{E_k\}_{k=1}^K \in \mathbb{R}^{K \times N \times D}$，其中 $E_k[i,:] \in \mathbb{R}^D$ 表示节点 $i$ 在基底 $k$ 下的嵌入。这种紧凑的基底集在捕获多样环境条件的同时鼓励跨节点复用，提升了分布偏移下的鲁棒性[^src-stpde]。

## 稀疏路由

给定节点潜在状态 $\mathbf{h}_i \in \mathbb{R}^D$，路由器 $R: \mathbb{R}^D \to \mathbb{R}^K$ 输出基底 logits，经稀疏 Top-K softmax 获得混合权重：

$$w_{i,k} = \frac{\exp([R(\mathbf{h}_i)]_k)}{\sum_{j \in \text{Top-}k(i)} \exp([R(\mathbf{h}_i)]_j)} \cdot \mathbb{I}(k \in \text{Top-}k(i))$$

节点环境嵌入通过加权聚合得到：

$$\mathbf{e}_i = \sum_{k=1}^K w_{i,k} \tilde{E}_k[i,:] \in \mathbb{R}^D$$

通过 AdaLN 调制扩散算子输出：

$$\tilde{\mathbf{h}}_i = \gamma(\mathbf{e}_i) \odot \mathbf{h}'_i + \beta(\mathbf{e}_i)$$

## 随机扰动（Stochastic Perturbation）

训练时为每个节点 $i$ 构建扰动基底 $\tilde{E}_k$，采用三种采样策略[^src-stpde]：

1. **原始基底** $E_k$：直接使用学到的基底
2. **专家平均** $\bar{e}$：可学习的共享向量，提供平滑先验
3. **置换基底** $\Pi(E_k)$：交换节点索引注入结构化错位

该机制扩展了训练期间的邻域多样性，是 OOD 泛化和跨城市迁移中鲁棒性的关键——消融实验中 OOD 下 w/o P 的退化最大[^src-stpde]。

## 负载均衡

为防止基底坍塌（所有路由指向少数基底），引入辅助损失：

$$\mathcal{L}_{\text{aux}} = K \sum_{k=1}^K P_k^2 - 1, \quad P_k = \frac{1}{|\mathcal{B}|N} \sum_{x \in \mathcal{B}} \sum_{i=1}^N w_{i,k}$$

当 $P_k = 1/K$ 时最小化，促进基底利用多样化[^src-stpde]。

## 在迁移和持续学习中的作用

- **跨城市迁移**：冻结 Invariant Diffusion Operator + 预训练基底，仅重建目标域的 Environment Basis Manifold（轻量微调）
- **持续学习**：新增量阶段仅扩展 Manifold 参数（$\Phi$ 和 $R$），冻结其余部分，显著减轻灾难性遗忘[^src-stpde]

[^src-stpde]: [[source-stpde]]
