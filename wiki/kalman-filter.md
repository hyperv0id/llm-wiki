---
title: "Kalman Filter"
type: concept
tags:
  - kalman-filter
  - state-estimation
  - dynamical-systems
  - uncertainty-quantification
created: 2026-06-08
last_updated: 2026-07-13
source_count: 2
confidence: medium
status: active
---

# Kalman Filter

> **编排者备注**：本页是从 [[k2vae|K²VAE]] ingest 中提议的基础概念页（仓库此前无 Kalman 滤波页面，但被 [[k2vae]] 与 [[kalmannet-uncertainty-modeling]] 引用）。Kalman 滤波是经典算法，建议作为共享概念页创建/去重。

**Kalman 滤波** (Welch & Bishop, 1995; Simon, 2001) 是估计线性动力系统状态的递归算法，分两步工作[^src-k2vae]：

1. **Predict（预测步）**：基于状态转移方程预测当前状态 $\hat{z}_k=Az_{k-1}+Bu_k$ 及不确定性协方差 $\hat{P}_k=AP_{k-1}A^T+Q$；
2. **Update（更新步）**：用观测与预测之差，按 **Kalman 增益** $K_k=\hat{P}_kH^T(H\hat{P}_kH^T+R)^{-1}$ 加权，精炼状态估计 $z_k=\hat{z}_k+K_k(o_k-H\hat{z}_k)$ 与协方差 $P_k=(I-K_kH)\hat{P}_k$。

Kalman 增益在"相信预测"与"相信观测"之间自适应权衡——观测噪声 $R$ 越小越相信观测，过程噪声 $Q$ 越小越相信预测。它有效融合多源信息以提升估计精度，同时**显式建模系统不确定性**[^src-k2vae]。

## 数值稳定性

更新步 $P_k=(I-K_kH)\hat{P}_k$ 在浮点运算下可能失去正定性。常用 **Joseph 形式** $P_k=(I-K_kH)\hat{P}_k(I-K_kH)^T+K_kRK_k^T$ 将其写为两个正定项之和以保持正定[^src-k2vae]。

## 在深度学习中的应用

经典 Kalman 滤波要求已知线性系统矩阵。深度学习中可把 $A,B,H$ 及噪声协方差 $Q,R$ 设为**可学习参数**，端到端训练——[[k2vae|K²VAE]] 的 [[kalmannet-uncertainty-modeling|KalmanNet]] 即是此类"神经化 Kalman 滤波"，在 Koopman 测量空间上迭代 Predict/Update，并把输出协方差对齐为 VAE 变分后验[^src-k2vae]。[[deep-state-space-model|深度状态空间模型]]路线更早的代表是 [[deepstate|DeepState]]（NeurIPS 2018）：用全局 RNN 从协变量输出线性高斯 SSM 的时变参数，训练与预测均调用解析 Kalman 滤波/平滑计算边际似然与潜状态后验[^src-deepstate][^src-k2vae]。

## 关联页面

- [[kalmannet-uncertainty-modeling]] — 神经化 Kalman 滤波用于不确定性建模
- [[deepstate]] — RNN 参数化线性 SSM + Kalman 似然/预测（NeurIPS 2018）
- [[deep-state-space-model]] — 深度状态空间模型概念
- [[k2vae]] — Koopman + Kalman 概率预测
- [[koopman-linearization-for-forecasting]] — Kalman 滤波作用的线性系统由 Koopman 线性化构造

[^src-k2vae]: [[source-k2vae]]
[^src-deepstate]: [[source-deepstate]]

