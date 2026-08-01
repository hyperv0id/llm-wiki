---
title: "Temporal Semantic Primitives"
type: technique
tags:
  - time-series
  - multimodal
  - llm
  - semantic-primitives
  - non-stationarity
  - tess
created: 2026-08-01
last_updated: 2026-08-01
source_count: 1
confidence: medium
status: active
---

# Temporal Semantic Primitives

**Temporal Semantic Primitives**（时间演化原语，TSP）是 [[tess|TESS]] 定义的四类离散语义类别，用于把文本对时间演化的隐式描述转成显式、定量、可验证的信号。设计来源是经典预测文献中的统计关键特征：前两类回答"时间序列的什么在演化"，第三、四类回答"影响何时显现、持续多久"[^src-tess]。

## 四类原语

### 1. Mean Shift（均值移位）

标准化均值差 $\Delta\mu = (\bar Y_t-\bar X_t)/\sigma(X_t)$，5 类：strong-rise / mild-rise / stable / mild-drop / strong-drop。

### 2. Volatility（波动率）

波动率对数比 $r_\sigma = \log[(\sigma_Y+\epsilon)/(\sigma_X+\epsilon)]$（一阶差分后 std），5 类：surge / rise / stable / fall / calm。

### 3. Shape（演化形态）

把预测窗口分成 $N_{fcst}$ 个 patch，取 patch 间均值差的符号序列 $s_i=\mathrm{sgn}_\tau(\bar u_{i+1}-\bar u_i)$ 的主导模式，5 类：ascend / descend / peak / trough / oscillate。

### 4. Lag and Decay（滞后与衰减）

基于影响分布 $\pi_i$（patch 级均值/波动偏移归一化）的三个指标：质心 $c$（onset 时机）、尾部质量 $d$（持续性）、峰值显著度 $q$（集中 vs 弥散），6 类：early-fade / early-persist / mid-fade / mid-persist / late / diffuse[^src-tess]。

## 关键性质：数值可验证性

给定观测与未来窗口，每个原语的真值 $v_{t,k}=\psi_k(Y_t)$ 唯一确定。这带来两个直接后果：

1. 门控网络无需人工标注即可获得监督：标签 $y_{t,k}=\mathbb{1}[\hat v_{t,k}=\psi_k(Y_t)]$（LLM 判对与否），BCE 训练。
2. 提取结果可审计：任何一条文本的原语预测都能对照真实序列核验[^src-tess]。

所有阈值（$\tau_1<\tau_2$、$\kappa_1<\kappa_2$、$\rho$、$\eta$）由训练集分位数自适应设定，避免跨数据集手工调参。

## 提取与使用

- **提取**：冻结 LLM 做小候选集多类分类。对每个候选类算 log-likelihood，温度缩放 softmax 得分布 $q_{t,k}$，argmax 得预测类。
- **不确定度**：top-1/top-2 log 概率 margin $m_{t,k}$ 作为校准信号，不额外采样。
- **注入**：预测类经可学习 embedding $E_k$ 映射为语义向量，乘门控 $g_{t,k}$ 后作 prefix token 与 patch embedding 拼接，全程参与自注意力[^src-tess]。

## 理论性质

- 错误传播：原语错误对预测的影响按 $g_{t,k}^2$ 衰减（forecaster 坐标 Lipschitz 假设下）。
- 统计效率：原语假设空间复杂度 $\sqrt{M/n}$（$M=\prod_k|V_k|$）优于 token 级 $\sqrt{\log|A_T|/n}$，当 $\prod_k M_k \ll T$ 时由 $\sqrt{T/n}$ 改进而来[^src-tess]。

## 相关页面

- [[source-tess]] · [[tess]] · [[non-fusion-guidance]] · [[patchtst]] · [[multimodal-time-series-forecasting]]

[^src-tess]: [[source-tess]]
