---
title: "Adaptive Conformal Inference"
type: concept
tags:
  - conformal-prediction
  - online-learning
  - uncertainty-quantification
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Adaptive Conformal Inference (ACI)

ACI 由 Gibbs & Candes（2021）提出，是一种纯误差反馈驱动的在线共形预测方法。核心思想是根据过去覆盖结果动态调整误差率 $\alpha_t$：当观测值落在区间外时增大 $\alpha_t$（扩大区间），反之减小。

## 变体
- **AgACI**（Zaffran et al., 2022）：通过多专家聚合解决参数敏感性
- **DtACI**（Gibbs & Candès, 2024）：改进的在线共形推理

## 问题
- 稳定期间过度优化 → 突变期间系统性覆盖不足
- 高区间方差
- 在 90% 目标下常低于所需边际覆盖率（86-87%）

[^src-sa-bcp]: [[source-sa-bcp]]
