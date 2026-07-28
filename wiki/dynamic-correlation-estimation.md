---
title: "Dynamic Correlation Estimation (DCE)"
type: technique
tags:
  - channel-correlation
  - low-rank-decomposition
  - time-series
  - tsfm-adapter
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# Dynamic Correlation Estimation (DCE)

**DCE** 是 [[cora-correlation-aware-adapter|CoRA (Correlation-aware Adapter)]] 中估计**动态通道相关矩阵**的模块：把可学习相关拆成低秩**时变**与**时不变**两部分，再叠规则 Pearson 先验，为后续异质/部分对比学习提供监督图[^src-cheng-2025-cora-correlation-aware-adapter]。

## 公式骨架

\[
M_t^{\mathrm{corr}} = R + Q_t V Q_t^\top
\]

- \(R\)：输入窗上的 Pearson 相关（规则项）  
- \(Q_t\in\mathbb{R}^{N\times M}\)：时变因子（\(M<N\)）  
- \(V\in\mathbb{R}^{M\times M}\)：全局时不变因子，\(V=\mathrm{Sigmoid}(\mathrm{ReLU}(E_1 E_2^\top))\)[^src-cheng-2025-cora-correlation-aware-adapter]

\(Q_t\) 由 **[[time-aware-polynomial-correlation|可学习时间多项式]]** 生成：共享基 \(q\) 的 Hadamard 幂次，系数 \(C_t=f(\tilde X_t)\) 由 TSFM 表示经 MLP 预测[^src-cheng-2025-cora-correlation-aware-adapter]。

## 设计意图

- **参数效率**：避免直接学满秩时变 \(N\times N\) 矩阵；Theorem 1 称在局部平稳下 \(Q_t V Q_t^\top\) 仍等价于「时不变 + 时变」加性分解[^src-cheng-2025-cora-correlation-aware-adapter]。  
- **显式动态规律**：多项式共享基跨时间，针对相关本身的趋势/周期，而非只堆 attention[^src-cheng-2025-cora-correlation-aware-adapter]。  
- **下游用途**：\(M^{\mathrm{corr}}\) 阈值成 \(M^{\mathrm{pos}}/M^{\mathrm{neg}}\)，驱动 [[heterogeneous-partial-contrastive-learning|HPCL]]；DCE 偏差会传导到对比标签[^src-cheng-2025-cora-correlation-aware-adapter]。

## 复杂度

训练含 Pearson 与 \(Q V Q^\top\) 组合，相对 \(N\) 为 \(O(N^2)\) 量级；**推理不跑 DCE**（CoRA 推理只保留投影）[^src-cheng-2025-cora-correlation-aware-adapter]。

## 相关页面

- [[cora-correlation-aware-adapter]] · [[source-cheng-2025-cora-correlation-aware-adapter]]
- [[time-aware-polynomial-correlation]] · [[heterogeneous-partial-contrastive-learning]]

## 引用

[^src-cheng-2025-cora-correlation-aware-adapter]: [[source-cheng-2025-cora-correlation-aware-adapter]]
