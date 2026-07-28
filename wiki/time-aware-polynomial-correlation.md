---
title: "Time-aware Polynomial Correlation"
type: technique
tags:
  - dynamic-correlation
  - polynomial
  - channel-correlation
  - tsfm-adapter
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# Time-aware Polynomial Correlation

**可学习时间多项式**是 [[dynamic-correlation-estimation|DCE]] 生成时变因子 \(Q_t\) 的方式：用共享矩阵基 \(q\) 的逐元素幂次叠加，系数随时间（由 TSFM 表示）变化，以拟合通道相关的趋势与周期[^src-cheng-2025-cora-correlation-aware-adapter]。

## 形式

\[
Q_t = \sum_{i=0}^{K} C_{i,t}\, q^{(i)},\quad q^{(i)}=\underbrace{q\odot\cdots\odot q}_{i\ \mathrm{times}}
\]

- \(q\in\mathbb{R}^{N\times M}\)：全局可学基（相关变化的模式）  
- \(C_t=(C_{0,t},\ldots,C_{K,t})=f(\tilde X_t)\)：由表示经 MLP 预测的通道-系数  
- 只学系数映射 \(f\)，不直接回归整块时变矩阵 → 轻量[^src-cheng-2025-cora-correlation-aware-adapter]

## 理论与实践

- **Theorem 2**：局部平稳且相关关于基光滑时，逼近误差随阶数 K 以 Maclaurin 余项下降[^src-cheng-2025-cora-correlation-aware-adapter]。  
- **敏感度**：K 较稳，常用 3 或 4；过大收益有限[^src-cheng-2025-cora-correlation-aware-adapter]。  
- 动机：既有动态图/相关方法缺少对「相关本身随时间规律」的显式函数化[^src-cheng-2025-cora-correlation-aware-adapter]。

## 相关页面

- [[dynamic-correlation-estimation]] · [[cora-correlation-aware-adapter]]
- [[source-cheng-2025-cora-correlation-aware-adapter]]

## 引用

[^src-cheng-2025-cora-correlation-aware-adapter]: [[source-cheng-2025-cora-correlation-aware-adapter]]
