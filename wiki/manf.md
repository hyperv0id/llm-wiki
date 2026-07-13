---
title: "MANF"
type: entity
tags:
  - time-series
  - probabilistic-forecasting
  - normalizing-flow
  - multi-scale-attention
  - non-autoregressive
  - realnvp
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# MANF（Multi-scale Attention Normalizing Flow）

**MANF** 是 Feng 等人提出的多变量**概率**时间序列预测模型（arXiv:2205.07493）：编码器以 [[multi-scale-attention|多尺度注意力]] + 动态相对位置建模层次时序结构，解码侧用条件 [[normalizing-flow|RealNVP 归一化流]] 非自回归地生成未来窗口的联合分布[^src-maf]。

## 问题设定

记多变量观测 $s_i$ 与协变量 $x_i$。训练将序列划分为历史窗 $[1,T)$ 与预测窗 $[T,T+k]$；模型学习条件分布 $p(s_{T:T+k} \mid s_{1:T-1}, x_{1:T+k})$，以 CRPS-sum / MSE 评估[^src-maf]。

## 架构

```
历史 s_{1:T-1} (+ 历史协变量嵌入)
        │
        ▼
  多尺度注意力编码器（尺度随层增大）
  H_1^enc = A(S, Θ_1) → … → H_l^enc
        │
        ▼  cross-attn 条件
未来协变量 x_{T:T+k} + PE ──► 解码器层 H_i^dec
        │
        ▼  逐层条件 affine coupling
z_0 ~ N(0,I) → z_1 → … → z_l = 预测样本
```

### 编码器：多尺度 + 相对位置
- 第 $l$ 层注意力窗口 $\Theta_l$ 增大，先局部后全局，利于日内/周内等层次结构[^src-maf]。
- 相对位置 $R_{x-y}$ 与尺度相关可学习 $u_{\Theta_l}, v_{\Theta_l}$ 进入注意力打分，避免“内容与位置混算”导致的位置失真[^src-maf]。
- 默认 $\Theta=[L/3,L/2,L]$，$L=4\times$ 预测长度；3 层编码[^src-maf]。

### 解码器 + 条件流（NAR）
- **不**把预测窗内真值 $s_t$ 回馈模型（相对 LSTM/Transformer-MAF 等 AR 流）[^src-maf]。
- 解码器用 vanilla self/cross-attention，输出 $H_i^{\mathrm{dec}}$ 作为 RealNVP 的 scale/translation 网络条件[^src-maf]：
  $$
  z_{i+1} = z_i \odot \exp\!\big(s(H_i^{\mathrm{dec}})\big) + t(H_i^{\mathrm{dec}}),\quad z_0\sim\mathcal{N}(0,I).
  $$
- 多层流对应多层解码条件：浅层“基础知识”、深层“强化知识”[^src-maf]。
- 训练最大化 batch 平均 log-likelihood；流中嵌入 BN 双射以稳训练[^src-maf]。

### 缩放与实现
- 各序列除以历史窗均值后输入，采样后再乘回（均值缩放）[^src-maf]。
- Adam，lr $5\times10^{-4}$ 或 $10^{-3}$，batch 64，约 60 epoch；流隐藏维 100，3 栈 bijection；评估 100 条样本经验 CDF[^src-maf]。
- PyTorch + GluonTS；训练用 mixup 增广[^src-maf]。

## 复杂度

多尺度注意力 $O(RTD)$（$R$ 为尺度窗口、$T$ 窗长、$D$ 维数），RealNVP $O(D)$；时间维上训练与测试均可并行，对比 AR Transformer/LSTM+flow 的 $O(D^2T)$[^src-maf]。Electricity 上每 epoch 训练约 2.2s、测试约 2.2s，显著快于 LSTM-MAF / Transformer-MAF[^src-maf]。

## 实证要点

| 数据集 | 维数 | MANF CRPS-sum（文中） | 相对强基线亮点 |
|--------|------|----------------------|----------------|
| Exchange | 8 | 0.004 | 弱周期，提升有限 |
| Solar | 137 | 0.253 | MSE $9.8\mathrm{e}2\to7.4\mathrm{e}2$ |
| Electricity | 370 | 0.014 | MSE $1.8\mathrm{e}5\to1.6\mathrm{e}5$ |
| Traffic | 963 | 0.026 | MSE $4.9\mathrm{e}{-4}\to4.1\mathrm{e}{-4}$ |
| Taxi | 1214 | 0.123 | 高随机下仍领先 AR 流 |
| Wikipedia | 2000 | 0.057 | MSE $3.8\mathrm{e}7\to2.87\mathrm{e}7$ |

加倍预测长度与缺失噪声压力测试中，MANF 相对 LSTM-MAF / Transformer-MAF 衰减更小，说明 one-shot 流有利于抗误差累积与可扩展性[^src-maf]。

## 消融

- **MANF-T**（编码器换 vanilla Transformer）：性能大幅下降 → 多尺度对时序相关关键[^src-maf]。
- **MANF-P**（多尺度 + 绝对位置）：介于 MANF 与 MANF-T 之间 → 相对位置与多尺度应一体[^src-maf]。
- **MANF-L**（仅末层条件进堆叠流）：仍优于多数基线，但逊于逐层条件流[^src-maf]。
- **MANF-M**（解码器也用多尺度）：与默认接近，故解码侧保留 vanilla attention[^src-maf]。

## 在谱系中的位置

- 相对 **AR 条件流**（LSTM-RealNVP / LSTM-MAF / Transformer-MAF）：MANF 用 NAR one-shot 生成，强调速度与抗累积误差[^src-maf]。
- 相对 **NKF**（LGM + flow）：不以线性高斯状态空间为骨架，直接用注意力序列建模 + 条件流[^src-maf]。
- 相对后续 **[[timegrad|TimeGrad]]**（AR + 扩散）：同属高维联合分布灵活建模，但生成机制与解码方案不同（流 vs 扩散，NAR vs AR）[^src-maf]。
- 属于 [[generative-time-series-forecasting|生成式时序预测]] 中**离散归一化流**一支，早于主流 Flow Matching / 扩散时序工作[^src-maf]。

## 局限与展望

弱周期数据收益有限；强随机需更深/更强流，但多层解码可能偏移；离散数据需 dequantization 或显式离散建模；未来可接 Flow++ 或非自回归扩散等更强生成器[^src-maf]。

## 相关页面

- [[source-maf]] — 源摘要
- [[multi-scale-attention]] — 多尺度注意力
- [[normalizing-flow]] — 归一化流
- [[ar-vs-nar-decoding]] — AR vs NAR
- [[generative-time-series-forecasting]]
- [[timegrad]] / [[tsflow]] / [[glow]]

[^src-maf]: [[source-maf]]
