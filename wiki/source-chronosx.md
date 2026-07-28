---
title: "ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables"
type: source-summary
tags:
  - time-series-foundation-model
  - covariate-adaptation
  - chronos
  - modular-adapters
  - aistats-2025
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# ChronosX 源文件摘要

**来源**: Sebastian Pineda Arango*, Pedro Mercado*, Shubham Kapoor, Abdul Fatir Ansari, Lorenzo Stella, Huibin Shen, Hugo Senetaire, Caner Turkmen, Oleksandr Shchur, Danielle C. Maddix, Michael Bohlke-Schneider, Yuyang Wang, Syama Rangapuram. *ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables.* AISTATS 2025 (PMLR 258). arXiv:2503.12107. Amazon Web Services / University of Freiburg / DTU. Code: `amazon-science/chronos-forecasting` (`chronosx` branch). raw: `raw/chronosx-adapting-pretrained-time-series-models-with-exogenous-variables.pdf`[^src-chronosx]

## 核心论点

多数预训练时序模型（[[chronos|Chronos]]、[[timesfm|TimesFM]]、MOMENT 等）在无协变量语料上训练，难以原生吃进任务特定外生变量；Moirai 的 any-variate attention 是例外，但“如何给**不原生支持协变量**的 TSFM 加协变量”仍开放[^src-chronosx]。ChronosX 借鉴 modular deep learning，用两个轻量适配块在**尽量少改预训练权重**的前提下注入 past/future 协变量。

## 机制：IIB + OIB

1. **Input Injection Block (IIB)**：对过去目标 token 嵌入 \(h_{\mathrm{emb}}(z_{t-1})\) 与 past 协变量 \(x_{t-1}\) 分别线性映射，拼接后 ReLU→FFN，**残差加回** token 嵌入再送入冻结/可训的 Chronos backbone（式 2–3）[^src-chronosx]。
2. **Output Injection Block (OIB)**：用 future 协变量与最后 hidden state，经同类 FFN 残差**修正 logits**（categorical token 分布；式 4–5）[^src-chronosx]。
3. **模块化变体**：可只用 past（IIB）、只用 future（OIB）或两者；**ChronosX** 默认两者且可冻结 backbone 只训适配器；**ChronosX(FF)** 全参微调[^src-chronosx]。
4. **跨 backbone**：同一框架扩展到 patch 输入与点预测——**TimesFMX**、**MOMENTX**（协变量同步 patch；点预测 OIB 在 \(\hat z_t\) 上加残差，式 6）[^src-chronosx]。

适配器默认两路独立线性 + 中间 ReLU 的 FFN；消融显示 One-Linear / No-Linear 等简化在多数设定弱于原设计[^src-chronosx]。

## 32 合成基准 + 18 真实集

为填补“可公开、可控动力学的协变量基准”缺口，作者构造 **32** 个合成集：主信号 4 类（Single / Simple / Diverse / Noisy sinusoids）× 协变量 4 类（spikes / steps / bells / ARP）× 组合算子 \(\{+,\times\}\)；每集 100 条日频序列、长度 1827、预测 30 步；Simple/Complex 各 16 集[^src-chronosx]。真实侧 **18** 集（ETT、M5、多电力市场、BDG-2、GEF、Rideshare 等），零售/能源/交通等，频率 15min–日，视界至约 30[^src-chronosx]。

## 主要结果

- 合成上，适配器-only 的 ChronosX / TimesFMX / MOMENTX 及 FF 变体相对各自零样本预训练显著吃进协变量；**ChronosX 相对 Chronos Small 约降 22% 聚合 WQL 与 MASE**[^src-chronosx]。
- 18 真实集上，ChronosX 在**已适配预训练模型**中 **WQL 最优**，MASE 进入前五；适配器-only 优于对应零样本；单序列数据上全量 FF 常不如强监督基线（TFT、DeepAR、PatchTSTx 等）[^src-chronosx]。
- **无协变量消融（NC）** 与 residual-only 对照表明增益主要来自协变量信息而非纯参数量；小 backbone（46M）对适配器-only 更稳，大 backbone 更利 FF[^src-chronosx]。

## 局限与定位

适配需下游训练，**失去纯零样本**；文中未做零初始化保护预训练嵌入空间。相对后续 [[unica|UniCA]] / [[cora-tsfm|CoRA]]，ChronosX 是早期 **past 前置 + future 后置 logits** 的模块适配范式，在 [[tsfm-covariate-adaptation-comparison|TSFM 协变量适配对比]] 中常被标为“无零初始化、偏前置”的对照基线[^src-chronosx]。

## 相关页面

- [[chronosx]] — 实体与机制
- [[chronos]] · [[timesfm]] · [[unica]] · [[cora-tsfm]] · [[source-cora]] · [[source-unica]]
- [[tsfm-covariate-adaptation-comparison]]

[^src-chronosx]: [[source-chronosx]]
