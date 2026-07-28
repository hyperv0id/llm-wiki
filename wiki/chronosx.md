---
title: "ChronosX"
type: entity
tags:
  - time-series-foundation-model
  - covariate-adaptation
  - modular-adapters
  - chronos
  - aistats-2025
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# ChronosX

**ChronosX** 是 Amazon 等提出的 **预训练时序模型协变量适配** 方法，AISTATS 2025 / arXiv:2503.12107。默认把 past/future 外生变量以两个轻量 **Injection Block** 接到 [[chronos|Chronos]] 上，并可同构扩展为 TimesFMX、MOMENTX[^src-chronosx]。

## 问题设定

预训练 TSFM 多在无协变量语料上训练；任务侧协变量维度与语义各异，难以直接进预训练。ChronosX 目标：在**不强制改写 backbone 接口**的前提下，用 modular adapters 吃进 \(X_{1:H}\)，建模 \(P(z_{C+1:H}\mid z_{1:C}, X_{1:H})\)[^src-chronosx]。

## 核心机制

### Input Injection Block（IIB，past）

对 token 嵌入与 past 协变量分别线性投影，拼接 → ReLU → FFN，**残差更新**嵌入后再进 Chronos 编码器：

\[
f_{\mathrm{IIB}}(z_{t-1}, x_{t-1}) = h_{\mathrm{emb}}(z_{t-1}) + g_{\mathrm{IIB}}(h_{\mathrm{emb}}(z_{t-1}), x_{t-1})
\]

属于 **编码器前 / 嵌入层** 注入，会改动进入 backbone 的表示分布[^src-chronosx]。

### Output Injection Block（OIB，future）

用 future 协变量与 last hidden 残差修正 **logits**（Chronos 的 categorical token 分布）；点预测骨干则在 \(\hat z_t\) 上加同类残差[^src-chronosx]。这是 **backbone 后 / 预测头侧** 注入。

### 训练模式

| 变体 | Backbone | 适配块 |
|------|----------|--------|
| ChronosX | 可冻结 | IIB+OIB |
| ChronosX(FF) | 全参微调 | IIB+OIB |
| ChronosX(IIB) / (OIB) | 可冻结 | 仅 past 或仅 future |
| TimesFMX / MOMENTX | 对应骨干 | 同框架 + patch / 点预测改写 |

适配器默认 **双线性 + FFN**；One-Linear / 去线性层消融在真实与 simple 合成上整体弱于原设计[^src-chronosx]。

## 合成 32 集基准

主信号（Single / Simple / Diverse / Noisy）× 协变量（spikes / steps / bells / ARP）× \(\{+,\times\}\) → 32 数据集；每集 100×1827 日频、pred=30。用于在可控动力学下测“是否真吃进协变量”[^src-chronosx]。

## 实证要点

- 合成：ChronosX 相对 Chronos Small **约 −22% 聚合 WQL/MASE**；适配器相对零样本骨干明显受益[^src-chronosx]。
- 真实 18 集：ChronosX 在适配后的预训练系中 **WQL 最佳**；适配器-only > 零样本；单序列集上全量 FF 常落后 TFT/DeepAR 等[^src-chronosx]。
- NC（去掉协变量支路）与 RS（仅线性残差协变量）消融支持：**增益主要来自协变量内容**[^src-chronosx]。

## 在适配谱系中的位置

在 [[tsfm-covariate-adaptation-comparison|TSFM 协变量适配全景]] 中，ChronosX 常被归为 **路线 A 前置注入** 代表（IIB 改嵌入），同时 OIB 提供后置 logits 修正——**混合注入、无零初始化**。后续 [[unica|UniCA]]（CAP 双融合、异构模态）与 [[cora-tsfm|CoRA]]（严格后置 adaLN + 零初始化 + 因果嵌入）在 CoRA 报告的 Sundial 公平对比中显著优于 ChronosX（avg MSE 0.134 vs CoRA 0.068）[^src-chronosx]；该对比数字来自 CoRA 侧评测，解读时需区分协议。

**局限（论文自述）**：适配训练牺牲零样本；未来方向包括类 in-context 的推理期协变量利用[^src-chronosx]。

## 相关页面

- [[source-chronosx]] — 源摘要
- [[chronos]] · [[timesfm]] · [[unica]] · [[cora-tsfm]] · [[source-cora]] · [[source-unica]]
- [[tsfm-covariate-adaptation-comparison]] · [[zero-initialized-adaptation]] · [[heterogeneous-covariates]]

[^src-chronosx]: [[source-chronosx]]
