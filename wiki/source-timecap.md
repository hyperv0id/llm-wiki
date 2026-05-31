---
title: "Source: TimeCAP"
type: source-summary
tags:
  - time-series
  - llm-agent
  - multimodal
  - event-prediction
  - contextualization
  - aaai2025
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# Source: TimeCAP

TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents, by Geon Lee, Wenchao Yu, Kijung Shin, Wei Cheng, Haifeng Chen (KAIST / NEC Labs), AAAI 2025 Oral[^src-timecap].

## 核心论点

现有 LLM-for-time-series 方法将 LLM 仅用作预测器（predictor），通过 fine-tuning 或 prompt tuning 将时间序列数据输入 LLM 获取预测。TimeCAP 的核心洞见是：*LLM 应该先理解上下文，再做预测*。真实世界的时间序列（气象、金融、医疗）背后有丰富的地理、气候、经济等上下文信息，直接给 LLM 喂原始数值等于浪费了 LLM 的领域知识和推理能力[^src-timecap]。

## 方法：双代理 + 多模态编码器

TimeCAP 包含三个核心组件：

1. **AC (Contextualizer Agent)**：用冻结的 GPT-4 将时间序列数据转换为文本摘要（textual summary），提取上下文信息（如"湿度上升伴随气压稳定，可能预示降雨前兆"）。这是纯零样本，无需训练[^src-timecap]。

2. **AP (Predictor Agent)**：基于 AC 生成的文本摘要进行事件预测。TimeCAP 的增强版本还会从训练集中检索相似摘要作为 in-context examples（prompt augmentation），帮助 AP 做出更准确的预测[^src-timecap]。

3. **Multi-Modal Encoder E_phi**：一个可训练的编码器，同时接受时间序列原始数据和 AC 生成的文本摘要（input augmentation）。架构：pretrained BERT 编码文本 + patch-based transformer encoder 编码时间序列，通过 multi-head self-attention 融合双模态。生成预测 y_MM 和 embedding z。最终预测是 AP 的输出和 encoder 输出的融合（λ-weighted linear combination）[^src-timecap]。

## 关键结果

- 在 7 个真实数据集上（天气 3 个 + 金融 2 个 + 医疗 2 个）全面超越 SOTA，平均 F1 提升 **28.75%**[^src-timecap]
- 仅用 contextualization（TimeCP）就达到 21.98% F1 提升，说明"理解上下文"的价值远超"直接预测"[^src-timecap]
- 零样本性能显著优于 PromptCast 和 LLMTime（同为零样本方法），因为 TimeCP 利用了 LLM 的领域知识来 contextualize[^src-timecap]
- 数据稀缺场景仍保持高性能：仅 10% 训练数据时性能下降远小于 PatchTST 和 GPT4TS[^src-timecap]
- LMaaS 兼容：所有 LLM agent 均冻结，通过 LMaaS API 调用，适用于黑盒 API[^src-timecap]
- 可解释性：提供了两种解释模式——隐式解释（生成预测理由）和显式解释（选出最相关的 in-context example）[^src-timecap]

## 局限性与讨论

- 依赖商业 LLM（GPT-4）API，成本和延迟是实际问题
- 训练部分（multi-modal encoder）需要 GPU，但推理时 LLM agent 无额外计算
- "contextualization"的质量取决于 LLM 对特定领域的理解深度，在某些专业领域可能不适用
- 仅处理分类任务（事件预测），未涉及回归预测

[^src-timecap]: [[source-timecap]]
