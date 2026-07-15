---
title: "Chronos"
type: entity
tags:
  - time-series
  - foundation-model
  - tokenizer
  - iclr2024
created: 2026-04-29
last_updated: 2026-07-17
source_count: 6
confidence: high
status: active
---

# Chronos

## 概述

**Chronos** 是 Amazon 提出的预训练时间序列语言模型，发表于 ICLR 2024[^src-chronos]。

## 核心贡献

1. **时间序列分词器 (Tokenizer)**：将连续时间序列量化为离散 token
2. **语言模型架构**：类似 T5 的编码器-解码器结构
3. **大规模预训练**：在多种领域的时间序列数据上训练
4. **零样本和少样本预测**：无需微调即可预测

## 架构

### Tokenizer

将时间序列通过以下步骤转换为离散 token：
1. **量化**：使用分位数将连续值离散化
2. **Binning**：将时间序列映射到固定数量的 bin
3. **Embedding**：将 bin ID 转换为向量

### 模型

- **架构**：类似 T5 的编码器-解码器 Transformer
- **规模**：~800M 参数（Chronos-Bolt）
- **Context Length**：2048 tokens

## 变体

| 变体 | 参数 | 特点 |
|------|------|------|
| Chronos-Tiny | 20M | 轻量级 |
| Chronos-Large | 200M | 高性能 |
| Chronos-Bolt | ~800M | 最新版本 |

## 与 UniCA 的结合

Chronos 是 UniCA 论文中验证的主要基础模型之一：

- **Chronos-Bolt + UniCA**：在单模态数据集上达到 0.506 MAPE（最优结果）
- **多模态任务**：Time-MMD 上达到 0.601 MAPE，提升 13%[^src-unca]

## 与 Sundial 的对比

Sundial (ICML 2025) 提出了与 Chronos 形成鲜明对比的设计选择[^src-sundial]:

- **Tokenization**: Sundial 使用连续 patch tokenization，而 Chronos 使用逐点离散 tokenization。Sundial 论证了离散 tokenization 导致 OOV 问题和粗粒度预测区间，并指出 Chronos 的逐点 tokenization 在长期预测中不如 patch 级预测
- **训练目标**: Sundial 使用 TimeFlow Loss (Flow Matching)，Chronos 使用 Cross-Entropy (下一 token 预测)
- **预训练规模**: Sundial 在 1032B 时间点上预训练 (约 11× Chronos 的 94B)
- **推理速度**: 在 FEV Leaderboard 上，Sundial 的推理速度较 Chronos 快 **35×** (Figure 5)，但 Sundial 零样本 MASE/WQL 指标仍排名第二（仅次于 Chronos）
- **概率预测**: 两者都支持概率预测，但 Sundial 通过生成式建模 (采样 20 预测) 实现，Chronos 通过离散 token 分布的采样实现

Sundial 团队在 Table 8 中报告：使用相同 94B 预训练数据子集时，Sundial 在 TSLib 上的零样本性能已超过 Chronos[^src-sundial]。

[[probts|ProbTS]] 进一步从统一点/分布基准指出：Chronos 通过量化 bin + Softmax 支持分布近似，但在高 [[non-gaussianity|非高斯性]] 短程场景相对领域专用概率模型（如 CSDI）的 CRPS 落差更大，预定义/离散分布头对复杂分布表达仍不足[^src-probts]。

## 相关页面

- [[unified-covariate-adaptation]] — UniCA 框架
- [[timesfm]] — TimesFM 基础模型
- [[timesnet]] — TimesNet 基础模型
- [[most]] — MoST 多模态时空基础模型（不同领域：TS vs ST）
- [[aurora]] — Aurora 多模态生成式基础模型（Chronos 为单模态，Aurora 支持多模态）
- [[tats]] — TaTS 即插即用多模态框架（Chronos 为数值专用 tokenization，TaTS 通过辅助变量处理文本）
- [[sundial]] — Sundial，连续 tokenization + Flow Matching 的对比模型 (ICML 2025)

## DynaMix 对比

[[dynamix|DynaMix]] (NeurIPS 2025) 对 Chronos 在 DSR 任务上进行了系统评估[^src-dynamix]：

- **长期动力学失败**：Chronos 在长期预测中倾向于收敛到不动点或简单周期轨道，无法重建混沌系统的真实吸引子几何[^src-dynamix]
- **Context Parroting**：Chronos 倾向重复性复现上下文模式而非适应系统的真实演化——功率谱呈现尖锐窄峰（暗示周期性），而真实混沌系统具有宽带弥散谱[^src-dynamix]
- **Lyapunov 指数**：Chronos 生成的轨迹 λ≈0.02（接近 0，指示周期行为），而真实 Lorenz-63 的 λ≈0.87[^src-dynamix]
- **参数量对比**：Chronos-t5-base（~200M）vs DynaMix（~10k），参数量高出约 20,000 倍，但 DSR 质量显著更差[^src-dynamix]
- **多变量耦合**：Chronos 将各维度独立处理，忽略了非线性动力系统中变量间的耦合——这是其 DSR 失败的根本原因之一[^src-dynamix]

然而，在底层系统真实为周期行为时，Chronos 能够正确捕获，DynaMix 亦然。短期预测上 Chronos 仍有一定竞争力[^src-dynamix]。
- [[timecap]] — TimeCAP LLM agent 框架，用 LLM 做时间序列上下文理解（AAAI 2025 Oral）

---

## 引用

[^src-chronos]: [[source-chronos]]
[^src-unca]: [[source-unca]]
[^src-sundial]: [[source-sundial]]
[^src-probts]: [[source-probts]]
[^src-dynamix]: [[source-dynamix]]