---
title: "Deep State Space Model"
type: concept
tags:
  - state-space-model
  - deep-learning
  - probabilistic-forecasting
  - kalman-filter
  - time-series
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Deep State Space Model（深度状态空间模型）

**深度状态空间模型**指用神经网络（常为 RNN）参数化或扩展经典[[kalman-filter|状态空间模型]]，使潜动态/观测模型既能编码趋势季节等结构先验，又能从大规模序列与协变量中联合学习的一类方法。NeurIPS 2018 的[[deepstate|DeepState]]是该范式在概率时间序列预测中的代表性实例：保留**线性高斯转移与观测**以便 Kalman 滤波解析推断，同时用全局 RNN 从协变量输出时变 SSM 参数。[^src-deepstate]

## 经典 SSM 与深度扩展的张力

- **经典 SSM**（ETS、结构时间序列、ARIMA 的状态空间形式）：可解释、小样本友好，但通常**每序列独立拟合**，难以共享模式，协变量与结构选择昂贵。[^src-deepstate]
- **纯深度序列模型**（如自回归 RNN）：跨序列特征学习强，但结构先验弱、可解释性差、对平滑等约束难施加。[^src-deepstate]
- **深度 SSM**：用网络生成或调制 SSM 参数/动态，试图同时获得结构与容量。[^src-deepstate]

## DeepState 式参数化（线性 + 全局 RNN）

DeepState 将映射写为 \(\Theta_t=\Psi(x_{1:t},\Phi)\)，其中 \(\Psi\) 由 LSTM 实现，\(\Theta_t\) 含转移矩阵、创新强度、发射向量与观测噪声等；数据似然为线性高斯 SSM 的边际 \(p_{\mathrm{SS}}\)，训练对共享 \(\Phi\) 最大化似然。非线性外生效应（促销等）主要通过协变量进入 \(\Psi\)，而非破坏线性潜动态。[^src-deepstate]

相关文献中的其他深度 SSM 变体包括：用 MLP 参数化转移的 Deep Markov Model、切断潜状态递归而依赖 RNN 隐状态的 Variational RNN / State-Space LSTM、以及保持线性高斯以便 Kalman 的 KVAE 等；DeepState 的特点是**直接让 RNN 输出完整 SSM 参数**，避免额外的局部线性基组合超参。[^src-deepstate]

## 与后续路线的关系

| 路线 | 与深度 SSM 的关系 |
|------|-------------------|
| DeepAR 类自回归似然 | 无显式潜状态转移；目标值作输入 |
| [[deepstate\|DeepState]] | 线性 SSM + RNN 参数头 + Kalman |
| [[k2vae\|K²VAE]] / [[kalmannet-uncertainty-modeling\|KalmanNet]] | Koopman 线性化 + 可学习 Kalman 精炼，VAE 一步生成 |
| 扩散/流概率模型（[[timegrad\|TimeGrad]] 等） | 弱结构先验、强分布表达，通常无解析 Kalman |

在[[generative-time-series-forecasting|生成式/概率时间序列预测]]谱系中，深度 SSM 构成“**可解释线性动态 + 神经参数化**”分支，与扩散、流匹配、归一化流并列。[^src-deepstate]

## 关联页面

- [[deepstate]] / [[source-deepstate]]
- [[kalman-filter]]
- [[k2vae]]
- [[generative-time-series-forecasting]]

[^src-deepstate]: [[source-deepstate]]
