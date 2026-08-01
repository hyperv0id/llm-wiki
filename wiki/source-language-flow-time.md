---
title: "Language in the Flow of Time: TaTS Framework"
type: source-summary
tags:
  - time-series
  - multimodal
  - text-time-series
  - forecasting
  - plug-and-play
created: 2026-06-18
last_updated: 2026-08-01
source_count: 1
confidence: medium
status: superseded
superseded_by: [[source-language-in-the-flow-of-time]]
---

# Language in the Flow of Time: TaTS

> [!warning] 已取代
> 本页与 [[source-language-in-the-flow-of-time]] 重复（同一论文 ICLR 2026, arXiv:2502.08942）；内容已合并至后者，本页保留存档。

## 概述

TaTS（Texts as Time Series）是 Li 等人提出的一个即插即用的多模态时间序列框架，发表于 ICLR 2026[^src-language-flow-time]。该框架将时间序列配对的文本转化为辅助变量，无缝集成到任何现有的数值型时间序列模型中，无需修改模型架构即可提升预测和插补性能。在多个基准数据集上，TaTS 在预测任务中取得了约 14% 的性能提升，在插补任务中提升高达 30%。

## 核心发现：时序文本共振

论文基于柏拉图表征假说（Platonic Representation Hypothesis），该假说认为描述同一对象的不同模态会收敛到共享的潜在表示空间[^src-language-flow-time]。将此假说扩展到时间序列，论文发现了一个关键现象——时序文本共振（Chronological Textual Resonance, CTR）：时间序列配对的文本嵌入在频域中表现出与对应数值序列高度相似的周期性模式。

具体而言，论文对 Economy、Social Good、Traffic 三个真实数据集进行了频率分析[^src-language-flow-time]。通过傅里叶变换分析时间序列的频谱，同时计算文本嵌入的 lag-similarity 并对其应用 FFT，发现配对文本的主要频率与时间序列的频率高度匹配。例如，月度采样的时间序列显示周期为 12（频率 0.083），配对文本也表现出相同的周期性。

CTR 现象的三个成因[^src-language-flow-time]：共享的外部驱动因素（如季节性变化、经济周期）；时间序列对文本的因果影响（文本响应数值趋势而演化，如新闻报道随经济指标更新）；以及文本中隐含的额外变量（如 GDP 报告中提及的股市指数）与时间序列周期对齐。

## TT-Wasserstein 度量

论文提出了 TT-Wasserstein，定义为时间序列和文本频谱之间的 Wasserstein 距离[^src-language-flow-time]。该度量量化了两者频谱对齐的程度。在 Time-MMD 数据集上，原始数据集的 TT-Wasserstein 远低于时间序列打乱或文本打乱的版本，验证了该度量的有效性。更重要的是，TT-Wasserstein 越低，TaTS 带来的性能提升越大——即该度量可以预测 TaTS 的潜在效果。

## TaTS 框架

TaTS 的工作流程极为简洁[^src-language-flow-time]：

1. 使用预训练语言模型（如 GPT-2，1.5B 参数）将每个时间戳的配对文本编码为嵌入向量（维度 d_text）。
2. 通过一个轻量级三层 MLP 将高维文本嵌入投影到低维空间（维度 d_mapped）。
3. 将投影后的文本表示作为辅助变量与原始时间序列沿变量维度拼接，形成增强的多变量序列 U ∈ R^{T × (N + d_mapped)}。
4. 将增强序列输入现有的时间序列模型进行下游任务，预测时仅提取前 N 个变量（原始时间序列）。

## 实验与消融

TaTS 与 iTransformer、PatchTST、FiLM、DLinear 等多种时间序列模型兼容[^src-language-flow-time]。在 Time-MMD、FNSPID 和 FNF 等数据集上，TaTS 一致且显著地提升了所有基线的性能。

消融实验揭示了多个关键发现[^src-language-flow-time]：不同文本编码器（BERT 110M、GPT-2 1.5B、LLaMA2 7B）均保持鲁棒性，更大模型带来轻微提升；打乱文本时间戳会破坏性能，甚至劣于纯数值基线；随机丢弃 25% 文本后 TaTS 仍有效；极端噪声文本可通过随机丢弃 40-80% 来缓解。TaTS 仅增加约 1% 的可学习参数和约 8% 的训练时间。替代融合架构（门控残差、交叉注意力）与 MLP 投影性能相当。

## 局限性

TaTS 的效果依赖于文本质量——当配对文本与时间序列无关时，TT-Wasserstein 接近随机水平，改进有限[^src-language-flow-time]。论文未深入探索更精细的多模态融合架构，也未充分研究文本编码器规模与效果的关系。

[^src-language-flow-time]: [[source-language-flow-time]]