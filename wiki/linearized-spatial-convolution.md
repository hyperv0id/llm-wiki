---
title: "Linearized Spatial Convolution (LSC)"
type: technique
tags:
  - graph-neural-network
  - linear-attention
  - traffic-forecasting
  - scalability
  - kernel-approximation
created: 2026-06-09
last_updated: 2026-06-09
source_count: 1
confidence: medium
status: active
---

# Linearized Spatial Convolution (LSC)

**线性化空间卷积 (LSC)** 是 [[bigst|BigST]] 提出的 **O(N)** 复杂度图消息传递算子，使自适应图 STGNN 可扩展到大规模路网[^src-bigst]。

## 问题
[[gwnet|GWNET]] 式自适应邻接 A=σ(E1E2ᵀ) 与逐跳特征传播 H⁽ᵏ⁾=AH⁽ᵏ⁻¹⁾W 需显式构造并存储 O(N²) 稠密邻接，对 N≈10 万的路网不可行（整体 O(TLN²)）[^src-bigst]。

## 方法：核分解 + 乘法重排
LSC 把邻接写成 A=D⁻¹A′，A′=exp(E1E2ᵀ/τ)，再借 Performer 的**正随机特征 (PRF)** 映射 φ 近似指数核：A≈D⁻¹φ(E1)φ(E2)ᵀ[^src-bigst]。设 Ê1=φ(E1)/√τ、Ê2=φ(E2)/√τ，则特征传播

H⁽ᵏ⁾ = D̂⁻¹ (Ê1 (Ê2ᵀ H⁽ᵏ⁻¹⁾)) W⁽ᵏ⁻¹⁾，  D̂ = diag(Ê1(Ê2ᵀ1_N))。

关键在**矩阵乘法结合律**：先算 Ê2ᵀH（r×d）再左乘 Ê1，**永不显式构造 N×N 邻接**，时间/空间复杂度降到 **O(N)**（具体 O(KNd²+Ndr) / O(KNd+Nr)，r 为随机特征维度）[^src-bigst]。另加距离先验空间正则 L_r=Σ −d_ij·log A_ij 约束学到的图结构[^src-bigst]。

## 关联
- [[bigst]] — 提出 LSC 的模型
- [[gwnet]] — LSC 所线性化的自适应邻接来源
- [[long-sequence-feature-extractor]] — BigST 时间维的对偶机制（同用 PRF 核线性化）
- [[linear-attention-unified-framework]] — 同源的核线性化注意力思想（此处用于空间维）
- [[large-scale-spatial-temporal-graph]] — LSC 解决的大规模可扩展性问题

[^src-bigst]: [[source-bigst]]
