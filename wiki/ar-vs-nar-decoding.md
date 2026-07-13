---
title: "AR vs NAR Decoding for Time-Series Forecasting"
type: concept
tags:
  - time-series
  - forecasting
  - autoregressive
  - decoding
  - foundation-model
created: 2026-07-13
last_updated: 2026-07-13
source_count: 3
confidence: medium
status: active
---

# AR vs NAR Decoding（时间序列预测中的自回归 / 非自回归解码）

**AR vs NAR decoding** 是 [[probts|ProbTS]] 显式区分的核心方法决策之一：给定历史 $x_{t-L:t}$ 与协变量，模型如何产生多步未来 $x_{t+1:t+T}$[^src-probts]。

## 形式化

设编码器 $f_\phi$ 产生隐状态 $h$，预测头 $p_\theta$ 输出点估计或条件分布采样[^src-probts]：

| 方案 | 编码 | 特点 |
|------|------|------|
| **Autoregressive (AR)** | $h_t = f_\phi(x_{t-1}, c_t, h_{t-1})$，逐步滚动 | 可利用已生成值捕获递推结构；误差沿 horizon 累积 |
| **Non-autoregressive (NAR)** | $h_{t+1:t+T} = f_\phi(x_{t-L:t}, c_{t-L:t+T})$ 一次生成 | 并行、无逐步反馈；需在单次前向中建模整段依赖 |

## 研究线偏好

- **长程点预测线**（Informer / Autoformer / [[patchtst|PatchTST]] / iTransformer / LTSF-Linear 等）几乎清一色采用 NAR，以规避长 horizon 误差累积[^src-probts]。
- **短程概率线**（[[deepar|DeepAR]] / [[timegrad|TimeGrad]] / 归一化流等 vs [[csdi|CSDI]] / 部分扩散方法）在 AR 与 NAR 之间更均衡[^src-probts][^src-deepar]。
- **时间序列基础模型**同样分裂：Lag-Llama / [[timesfm|TimesFM]] / Timer / [[chronos|Chronos]] 偏 AR；MOIRAI / UniTS / ForecastPFN / TTM 偏 NAR[^src-probts]。

- **[[manf|MANF]]（arXiv 2022）** 给出早期 **NAR + 精确似然流** 证据：相对 LSTM-MAF / Transformer-MAF，加倍预测长度与缺失噪声下 CRPS/MSE 衰减更小，并显著加速训练/测试，支持“one-shot 生成可抑制 AR 误差累积”的论断[^src-maf]。

## ProbTS 实证规律

1. **AR 长程痛点是误差累积**：原始归一化下，AR 概率模型（如 TimeGrad）的 CRPS 随 horizon 与趋势强度上升而恶化[^src-probts]。
2. **AR 在强季节性上可占优**：Traffic 等强季节场景中 AR 可超过 PatchTST 等 NAR 点模型；季节越强，AR 相对优势越大[^src-probts]。
3. **归一化可部分“复活”长程 AR**：[[instance-normalization|RevIN]] 显著改善多数数据集上的 AR 长程表现（推测主要通过对冲趋势引起的分布漂移），但对强季节弱趋势的 Traffic 有负作用[^src-probts]。
4. **NAR 概率也非免费午餐**：CSDI 类 NAR 扩散在长 horizon 面临显存与学习效率问题，小数据集上长程表现亦欠佳[^src-probts]。
5. **TSFM 复现同一张力**：短 horizon 上 AR 基础模型有竞争力；horizon 拉长后 NAR（如 MOIRAI）优势扩大[^src-probts]。

## 开放问题

- 如何在保留 AR 对季节/递推结构建模优势的同时系统抑制长程误差累积[^src-probts]；
- 如何设计对长程概率预测既高效又表达充分的 NAR 架构与归一化[^src-probts]；
- 基础模型是否需要 horizon 自适应的混合解码策略[^src-probts]。

## 相关页面

- [[probts]] / [[source-probts]]
- [[generative-style-decoder]] — Informer 的非自回归一次前向解码
- [[timegrad]] — AR + 条件扩散
- [[deepar]] — 全局共享 AR-RNN + 参数化似然
- [[csdi]] — NAR 条件扩散（插补/预测）
- [[timesfm]] / [[chronos]] — AR 系基础模型
- [[instance-normalization]]
- [[generative-time-series-forecasting]]
- [[manf]] — NAR + 条件 RealNVP
- [[multi-scale-attention]] — MANF 编码器技术

[^src-probts]: [[source-probts]]
[^src-maf]: [[source-maf]]
[^src-deepar]: [[source-deepar]]
