---
title: "Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion"
type: source-summary
tags:
  - multimodal-fusion
  - schrodinger-bridge
  - rectified-flow
  - reliability-assessment
  - geometric-deep-learning
  - evidential-learning
created: 2026-07-18
last_updated: 2026-07-18
source_count: 0
confidence: medium
status: active
---

# Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion

**GMF** (Geometry-based Multimodal Fusion) 是一种将潜在空间传输几何用于多模态融合可靠性评估的框架。论文的核心论点是：可信的多模态融合应将可靠性视为**外在几何属性**而非内在预测结果，从而打破传统统计方法中"用分类器置信度检测错误"的循环依赖。

## 核心贡献

1. **几何可靠性评估**：通过 Diffusion Schrödinger Bridge (DSB) / Rectified Flow 在潜在空间中学习模态内（intra-modal）和模态间（inter-modal）传输代价。模态内传输能量 $E_{\text{intra}}$ 度量单个模态特征偏离流形结构的程度；模态间传输代价 $E_{\text{inter}}$ 度量跨模态语义一致性。两者均不依赖分类器输出。

2. **熵正则化最优融合权重**：将融合权重形式化为熵正则化几何代价最小化问题，其闭式解为 Gibbs 分布 $w^{*(m)} \propto \exp(-C^{(m)}/\tau)$，其中有效几何代价 $C^{(m)} = E_{\text{intra}}^{(m)} - \tau \ln \tilde{\gamma}_{\text{int}}^{(m)}$ 综合了内在质量和外在支持。

3. **几何屏障原理**（Theorem 4.5）：当模态编码冲突类别时，跨模态传输代价存在严格正下界 $\geq (\delta - 2\epsilon)^2$，导致冲突模态的融合权重被指数级抑制（Corollary 4.6）。

4. **梯度分离训练**：几何分支（速度网络）和决策分支（evidential 分类器）通过分离的梯度路径独立更新，避免几何信号被分类目标污染。

## 实验

在四个基准上验证：NYU Depth V2 (RGB-D)、UPMC Food-101 (Image-Text)、MVSA-Single (Image-Text 情感分析)、PneumoniaMNIST (X-ray/Report 医学诊断)。涵盖传感器噪声、语义冲突、不完整模态三种压力测试。GMF 在所有低质量场景下均优于 QMF、PDF、DBF、UAW-EEF、GOMFuNet 等基线，尤其在语义冲突检测上 Safe Rejection Rate 达 76.8% (vs 最佳基线 35.2%)。

## 局限

依赖上游编码器的表示质量（Assumption 4.1 的流形分离假设）。若特征提取器发生 mode collapse，几何传输代价可能失去辨别力。此外，速度网络引入了轻量额外参数，但推理开销可忽略（单步推理 ~18ms，与 PDF/DBF 等轻量基线持平）。
