---
title: "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - transformer
  - non-stationarity
  - cointegration
  - icml-2025
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting

- **作者/机构**：Peiyuan Liu\*、Beiliang Wu\*、Yifan Hu\*、Naiqi Li、Tao Dai、Jigang Bao、Shu-tao Xia（清华大学深圳国际研究生院 / 深圳大学）[^src-timebridge]
- **出处**：ICML 2025（PMLR 267）[^src-timebridge]
- **分级**：Tier 1（建立机制页 [[timebridge]]，未为内部组件建独立页）

现有预测方法对待非平稳性常走两个极端：用 RevIN 等手段全局去平稳会抹平跨变量共同趋势，全量保留又会在单变量短程窗内诱发伪相关。[[timebridge|TimeBridge]] 将非平稳性按作用域正交拆解：先在变量内用去趋势稳态分量做时间注意力以抑制短程伪相关、同时保留原始值免去反归一化，再压缩时间维后于保留非平稳性的前提下做跨变量注意力以捕获长期协整[^src-timebridge]。

## 去黑话算子配方与图谱关系

模型由标准算子按两阶段串行构成：先将输入 $X\in\mathbb{R}^{C\times I}$ 切分为 patch $P\in\mathbb{R}^{C\times N\times D}$（$N=I/S$），以 `Pad + AvgPool` 作平滑差得 $P'$；变量内时间注意力（Integrated Attention）令 $Q,K$ 取稳态 $P'$、$V$ 取未去趋势 $P$；随后 MLP 将块数 $N$ 压至 $M$，在同时间槽跨变量切片上直接计算非平稳注意力（Cointegrated Attention），经 `Flatten + Linear` 输出 $\hat{Y}\in\mathbb{R}^{C\times O}$[^src-timebridge]。相对 [[patchtst|PatchTST]] 属瓶颈接力 $\Delta$（加入 $Q/K/V$ 非对称去趋势与跨变量协作）；相对 [[itransformer|iTransformer]] 属同槽位互斥（$M$ 压缩时间槽交互 vs 单全局 variate token）；其非对称 $Q/K$ 注意力被 [[source-pesd-tsf|PESD-TSF]] 直接继承，时频加权 MAE 损失被 [[source-renf|ReNF]] 沿用[^src-timebridge]。

## 证据

1. **主结论与归因**：9 项 LTSF 8 项最优（较 [[timemixer|TimeMixer]] 降 MSE $13.66\%$），PeMS 亦领先；最大掉点来自放反去平稳域（Traffic $0.360\to 0.381$）或颠倒阶段（$0.360\to 0.396$），单独去趋势贡献仅 $0.001\text{–}0.02$[^src-timebridge]。
2. **缺口与负结果**：ETTh2 输 ModernTCN（$0.341$ vs $0.322$）与 PatchTST，金融 ARR 升但波动/回撤更差；缺正文协整定理、固定 $L=720$ 口径且无 Limitations 节[^src-timebridge]。

## 相关页面

- [[timebridge]] — 模型机制页（含分阶段维度流）
- [[patchtst]] · [[itransformer]] · [[timemixer]] — 核心基线与架构对照
- [[channel-independence]] · [[instance-normalization]] · [[dataset-stationarity-estimation]] — 通道与非平稳性处理谱系
- [[source-pesd-tsf]] · [[source-renf]] — 后续承接与损失/效率对照

[^src-timebridge]: [[source-timebridge]]
