---
title: "TRACE: Grounding Time Series in Context for Multimodal Embedding and Retrieval"
type: source-summary
tags:
  - time-series
  - multimodal
  - retrieval
  - cross-modal
  - contrastive-learning
  - neurips-2025
created: 2026-08-19
last_updated: 2026-08-19
source_count: 0
confidence: low
status: active
---

# TRACE: Grounding Time Series in Context for Multimodal Embedding and Retrieval

**Authors**: Jialin Chen, Ziyu Zhao, Gaukhar Nurbek, Aosong Feng, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, Rex Ying (Yale / McGill / UTRGV). NeurIPS 2025. Code: `github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder`.

## 核心贡献

TRACE 是首个多模态时序检索器（multimodal time-series retriever），通过将时间序列嵌入与对齐的文本上下文进行语义锚定（semantic grounding），实现跨模态检索（Text→TS / TS→Text / TS→TS）。此前时序检索方法均仅使用单模态时序嵌入，不纳入文本信号。TRACE 同时作为通用检索器和独立编码器服务下游预测与分类任务。

## 两阶段训练

1. **Stage 1 — 时序编码器预训练**：Encoder-only Transformer + [[channel-biased-attention|Channel-biased Attention (CbA)]] + [[channel-identity-token|Channel Identity Tokens (CITs)]] + RoPE（通道内独立施加）+ [[patch-based-tokenization|patch tokenization]]。掩码重建目标（MSE loss，掩码率 γ），token 级与通道级双重学习。使用 RevIN 预处理。

2. **Stage 2 — 跨模态对齐**：冻结时序编码器，用冻结 Sentence-Transformer（nomic）编码文本，通过 [[dual-level-hard-negative-mining|双级硬负采样]]（sample-level + channel-level）执行双向 InfoNCE 对齐。channel-level 对齐每个通道 CIT 嵌入与对应通道级文本描述；sample-level 对齐 [CLS] 嵌入与样本级上下文文本。λ_ch=1.0 默认。

## RAG 框架

给定查询时序 Xq，计算 [CLS] 嵌入 hq，用余弦相似度检索 top-R 多模态对 (Xi, τi_cxt)。每个检索对经可训练线性投影压缩为 soft prompt P，前置于冻结 TSFM 输入。仅 Proj 和 Head 可训练，骨干冻结。支持 Time-MoE / Timer-XL / Moment 三种 TSFM。

## 数据集

- 自建天气多模态数据集（扩展 MTBench）：NOAA Storm Events 事件报告 + GHCN-h 气象时序，74,337 实例，7 通道（温度/降水/湿度/能见度/风u/风v/天况），3 种时间分辨率。ChatGPT 生成通道级文本描述。
- [[time-mmd|Time-MMD]] 公开数据集（Health/Energy/Environment）用于跨域预测验证。

## 实验结果

- **检索**：Text→TS 和 TS→Text 检索精度 SOTA，nomic 文本编码器优于 bge/MiniLM，cross-attention 模块在小 K 下关键。
- **RAG**：分类准确率 +4.56%，预测误差 −4.55%。decoder-only 模型对检索模态丰富度更敏感，TS+Text > TS-only。
- **独立编码器**：预测任务在 Weather/Health/Energy/Environment 上超越 DLinear/PatchTST/iTransformer/TimesNet/TimeMixer/FSCA 及多个 TSFM（Chronos/Time-MoE/TimesFM/Timer-XL/Moment/Moirai）。分类任务（9 类天气事件）TRACE 89.76% Acc / 72.36% F1（+RAG），无 RAG 85.20% / 69.98%，超越所有基线。
- **效率**：10.78M 总参数，微调仅激活 0.12M 参数（~200× 少于 FSCA，~700× 少于 Time-MoE_small），训练 6.054s/epoch。

## 局限性

依赖训练时可用对齐的 TS–文本对，部分领域对齐可能噪声或缺失；channel-level 对齐引入训练计算开销增量。未来工作：弱/半监督、域适应、自回归条件生成。
