---
title: "MAGE: Less but More — Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting"
type: source-summary
tags:
  - spatiotemporal-forecasting
  - graph-learning
  - adaptive-graph
  - mixture-of-experts
  - linear-complexity
  - traffic-forecasting
  - kernel-method
created: 2026-07-19
last_updated: 2026-07-19
source_count: 0
confidence: medium
status: active
---

# MAGE — Source Summary

**Authors**: Jiaming Ma, Binwu Wang (corresponding), Guanjun Wang, Kuo Yang, Zhengyang Zhou, Pengkun Wang, Xu Wang, Yang Wang (corresponding) — USTC / Suzhou Institute for Advanced Research, USTC
**Venue**: NeurIPS 2025
**Code**: [official repository](https://github.com/JiamingMa666/MAGE)

## 核心贡献

MAGE (Mixture of Adaptive Graph Experts) 针对 STGNN 中自适应图学习的两个根本问题：**(1) ReLU 激活放大边级噪声**（负边权被放大，正边权被抑制，强化伪相关）[^src-mage]，**(2) O(N²dG) 二次复杂度**限制大规模扩展[^src-mage]。

MAGE 的三层创新：

1. **线性自适应图学习**：用 kernel-based 近似（指数激活函数前置于内积之前，Φ(e₁)=exp(e₁+η), Ψ(e₂)=exp(e₂+ξ)），利用乘法结合律将图卷积从 O(N²dG) 降为 O(N·d·dG)，线性于节点数[^src-mage]。代价是低秩瓶颈——Rank(A) ≤ dG ≪ N[^src-mage]。

2. **稀疏平衡 MoE**：KG 个候选专家各学不同自适应图，每节点 Top-K 稀疏激活 + 负载均衡 β 调制（符号 SGD 优化），将表示矩阵秩从 ≤dG 提升至 ≤KdG，当 K≥⌈d/dG⌉ 时恢复满秩表达能力[^src-mage]。每个专家用差分图学习：A(k) = Softmax(E₁)Softmax(E₂ᵀ) − λ·Softmax(E₃)Softmax(E₄ᵀ)，λ 可学[^src-mage]。

3. **理论洞察**：(a) ReLU 噪声放大理论——ReLU 使负相似度转化为正边权，强化伪空间依赖[^src-mage]；(b) MAGE 单步图卷积等价于传统图的多步卷积[^src-mage]。

## 实验

17 个数据集（Traffic/Energy/Meteorology/Mobile 四域）、14 个基线（AGCRN/GWNet/D²STGNN/PatchSTG/STAEformer/BigST/GSNet 等），MAGE 在 94%(48/51) 指标上 SOTA[^src-mage]。计算效率上，比 D²STGNN 快 118–960×、比 PatchSTG 快 4.7×、显存省 1.72×[^src-mage]。仅 3 层 MAGE Block + hyper residual connection，所有实验统一配置（d=128, dG=32, KG=16, K=4）[^src-mage]。

[^src-mage]: [[source-mage]]
