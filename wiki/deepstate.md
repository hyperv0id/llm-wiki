---
title: "DeepState"
type: entity
tags:
  - state-space-model
  - probabilistic-forecasting
  - time-series
  - kalman-filter
  - rnn
  - neurips-2018
  - amazon
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# DeepState

**DeepState**（Deep State Space Models for Time Series Forecasting）是 Amazon Research 提出的概率时间序列预测模型，发表于 NeurIPS 2018。它用全局共享的 LSTM 从协变量序列生成每条时间序列上**线性高斯状态空间模型**的时变参数，再用[[kalman-filter|Kalman 滤波]]计算边际似然与多步预测后验，从而在可解释 SSM 结构与深度联合学习之间取得折中。[^src-deepstate]

## 定位

| 维度 | DeepState |
|------|-----------|
| 内生建模 | 线性 SSM（水平/趋势/季节潜状态） |
| 外生/协变量 | 时间与静态特征经 RNN 注入 \(\Theta_t\) |
| 跨序列学习 | 共享网络参数 \(\Phi\)，非每序列独立拟合 |
| 似然 | 线性高斯，Kalman 解析 |
| 预测 | 潜状态后验 + 递推采样，非目标值自回归输入 |
| 代表对照 | ETS/ARIMA（经典 SSM）、DeepAR（自回归 RNN）、MatFact |

DeepState 属于[[deep-state-space-model|深度状态空间模型]]早期工业实践，也是后续神经 Kalman / Koopman-Kalman 路线（如[[k2vae|K²VAE]]）的重要前驱对照。[^src-deepstate]

## 核心机制（摘要）

1. **RNN 参数头**：\(h_t=\mathrm{LSTM}(h_{t-1},x_t;\Phi)\)，映射到 \((F_t,g_t,a_t,b_t,\sigma_t,\ldots)\)。[^src-deepstate]
2. **训练**：最大化 \(\sum_i\log p_{\mathrm{SS}}(z^{(i)}_{1:T_i}\mid\Theta^{(i)}_{1:T_i})\)，\(p_{\mathrm{SS}}\) 由 Kalman 滤波实现并可反传。[^src-deepstate]
3. **推理**：训练窗得 \(p(\ell_T\mid z_{1:T})\)；预测窗只 unroll 一次 RNN，再采样 \(K\) 条未来轨迹。[^src-deepstate]

## 经验要点

- 小样本 electricity/traffic（2–4 周训练）上相对 auto.arima、ets、DeepAR 的 p50/p90 优势，体现显式季节 SSM 结构的数据效率。[^src-deepstate]
- 合成实验表明联合训练可逐步恢复 day-of-week 季节模型的 \(\mu_0\)、创新与观测噪声参数。[^src-deepstate]
- 目标值不进入网络输入 → 缺失处理与多样本路径生成更高效。[^src-deepstate]

## 关联页面

- [[source-deepstate]] — 源摘要
- [[deep-state-space-model]] — 概念抽象
- [[kalman-filter]] — 推断算法
- [[generative-time-series-forecasting]] — 概率预测大图
- [[k2vae]] / [[kalmannet-uncertainty-modeling]] — 现代神经 Kalman 延伸

[^src-deepstate]: [[source-deepstate]]
