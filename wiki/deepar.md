---
title: "DeepAR"
type: entity
tags:
  - probabilistic-forecasting
  - time-series
  - autoregressive
  - rnn
  - negative-binomial
  - amazon
  - arxiv-1704-04110
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# DeepAR

**DeepAR**（Probabilistic Forecasting with Autoregressive Recurrent Networks）是 Amazon Research 提出的**全局共享自回归 RNN** 概率时间序列模型（Salinas, Flunkert & Gasthaus；arXiv:1704.04110，v3 2019）。它在大规模相关序列上联合训练 LSTM，输出每步似然参数，并通过 Monte Carlo 样本路径给出校准的分位数与区间预测，成为零售需求、电力、交通等场景中长期对照的工业基线。[^src-deepar]

## 定位

| 维度 | DeepAR |
|------|--------|
| 内生建模 | 多层 LSTM 自回归（\(z_{t-1},h_{t-1}\) 输入） |
| 跨序列学习 | 全局共享参数 \(\Theta\)，非每序列独立拟合 |
| 似然 | 高斯（实值）/ 负二项（正计数）；可扩展 |
| 预测 | 祖先采样轨迹 → 任意跨度分位数 |
| 尺度 | \(\nu_i\) 缩放 + 按速度加权采样 |
| 代表对照 | ARIMA/ETS/ISSM、MatFact、后续 [[deepstate\|DeepState]] / [[timegrad\|TimeGrad]] / [[tft\|TFT]] |

DeepAR 属于[[generative-time-series-forecasting|概率时间序列预测]]中的 **AR + 参数化似然** 早期深度代表，也是后续“RNN 记时间、更灵活头建模分布”路线（如 TimeGrad）的直接前驱。[^src-deepar]

## 核心机制（摘要）

1. **共享编码器–解码器 LSTM**：条件窗与预测窗同结构共权重；\(h_t=h(h_{t-1},z_{t-1},x_t)\)，\(\theta_t=\theta(h_t)\)。[^src-deepar]
2. **训练**：滑动窗最大化 \(\sum\log\ell(z_t\mid\theta_t)\)；teacher-forcing；计数数据用负二项。[^src-deepar]
3. **尺度**：自回归输入除以 \(\nu_i\)，尺度相关似然参数乘回；训练按 \(\nu_i\) 非均匀采样，缓解幂律销量 skew。[^src-deepar]
4. **推理**：条件窗喂入历史，预测窗采样 \(\tilde z_t\sim\ell(\cdot\mid\theta_t)\) 并回馈，重复得轨迹集合。[^src-deepar]

## 经验要点

- 相对 Croston、ETS、Snyder、ISSM 等，在 `parts` 与 Amazon `ec`/`ec-sub` 上 0.5/0.9-risk 整体约 15% 量级改进（相对最强已发表基线归一化）。[^src-deepar]
- `electricity`/`traffic` 上 ND/RMSE 优于矩阵分解 MatFact。[^src-deepar]
- 负二项 + 缩放/加权采样消融证明：计数似然与 power-law 处理均关键。[^src-deepar]
- 不确定性增长与季节模式可从数据学出，长跨度校准依赖轨迹时间相关（打乱后变差）。[^src-deepar]
- 跨序列共享支持**新品/短历史**序列，协变量（age、日历、类目嵌入）编码组依赖。[^src-deepar]

## 后续影响

- [[deepstate|DeepState]] 刻意**不把目标值喂入网络**，改用 RNN 参数化线性 SSM + Kalman，以换取缺失鲁棒与采样效率。[^src-deepar]
- [[timegrad|TimeGrad]] 保留 AR-RNN 与均值缩放，用条件扩散替换高斯/负二项输出头。[^src-deepar]
- [[probts|ProbTS]] / [[ar-vs-nar-decoding|AR vs NAR]] 将 DeepAR 式 AR 概率模型定位为短程分布强、长程易误差累积的代表轴。[^src-deepar]
- [[tft|TFT]]、[[tide|TiDE]] 等在多 horizon / 协变量设定中常以 DeepAR 为强基线。[^src-deepar]

## 关联页面

- [[source-deepar]] — 源摘要
- [[generative-time-series-forecasting]] — 概率预测大图
- [[ar-vs-nar-decoding]] — 解码方案
- [[deepstate]] / [[deep-state-space-model]] — SSM 对照
- [[timegrad]] — 扩散后继
- [[probts]] — 统一基准中的 AR 概率代表

[^src-deepar]: [[source-deepar]]
