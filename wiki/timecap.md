---
title: "TimeCAP"
type: entity
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
confidence: medium
status: active
---

# TimeCAP

TimeCAP (Contextualize, Augment, and Predict) 是一个利用 LLM agents 进行时间序列事件预测的框架，由 KAIST 和 NEC Labs 合作开发，发表于 AAAI 2025 Oral[^src-timecap]。

## 核心创新

TimeCAP 的核心创新是将 LLM 的角色从"预测器"扩展到"上下文理解器"。传统 LLM-for-time-series 方法直接用原始数值或参数化 embedding 输入 LLM 做预测，但数值数据与 LLM 预训练时的文本数据存在本质差异，导致 LLM 无法发挥其语义理解优势[^src-timecap]。

TimeCAP 包含两个独立的 LLM agent：

- **AC (Contextualizer Agent)**：利用 LLM 的领域知识，将时间序列数据转化为自然语言文本摘要（contextualization）[^src-timecap]
- **AP (Predictor Agent)**：基于 AC 生成的文本摘要进行事件预测[^src-timecap]

两者通过一个可训练的 **多模态编码器 (Multi-Modal Encoder)** 相互增强：

- **输入增强 (Input Augmentation)**：AC 生成的文本摘要作为 encoder 的额外输入，与时间序列数据融合[^src-timecap]
- **提示增强 (Prompt Augmentation)**：encoder 从训练集中检索相似文本摘要，作为 in-context examples 提供给 AP[^src-timecap]

## 架构细节

```
Time Series x ──→ AC (GPT-4) ──→ Text Summary sx
                    │                     │
                    │ Input Aug           │
                    ▼                     ▼
              Multi-Modal Encoder E_ϕ ◄───┘
              (BERT + Transformer)
                    │                     │
                    │                     │ Prompt Aug (k-NN retrieval)
                    ▼                     ▼
              Prediction y_MM     AP (GPT-4) + In-Context Examples
                    │                     │
                    └── Fused ────────────┘
                         ŷ = λ·y_LLM + (1-λ)·y_MM
```

Multi-modal encoder 的文本支路使用预训练 BERT（小模型，可微调），时间序列支路使用 patch-based transformer 编码器，通过 multi-head self-attention 融合双模态。最终预测为 AP（LLM）和 encoder 输出的线性融合[^src-timecap]。

## 性能亮点

在 7 个真实数据集（天气×3、金融×2、医疗×2）上的实验结果：

| 指标 | 数值 |
|------|------|
| 平均 F1 提升（vs SOTA） | 28.75% |
| 仅 contextualization 提升（TimeCP） | 21.98% |
| 零样本性能（vs PromptCast） | 显著领先 |
| 数据稀缺（10% 训练数据） | 性能下降远小于 competitors |

具体数据集表现：天气 NY 上 F1=0.676、医疗 MT 上 F1=0.947（远超第二名 PatchTST 的 0.695）、金融 SP 上 F1=0.398（对比 PatchTST 0.373）[^src-timecap]。

## 关键特性

- **LMaaS 兼容**：所有 LLM agent 均冻结，通过黑盒 API 调用，无需访问模型内部参数[^src-timecap]
- **可解释性**：提供隐式解释（生成预测理由）和显式解释（指出最相关 in-context example）两种模式[^src-timecap]
- **少数据友好**：在仅 10% 训练数据时仍保持高性能，零样本场景下优于所有 zero-shot 方法[^src-timecap]
- **数据集贡献**：论文公开了 7 个数据集及 LLM 生成的文本摘要，为后续研究提供基准[^src-timecap]

## 与前驱方法的关系

| 方法 | LLM 角色 | 上下文理解 | 文本增强 |
|------|----------|-----------|---------|
| PromptCast | Predictor | 仅基础文本化 | 无 |
| LLMTime | Predictor | 无 | 无 |
| Time-LLM | Predictor (via reprogramming) | 无 | 无 |
| GPT4TS | Predictor (fine-tuned) | 隐含 | 无 |
| **TimeCP** | Contextualizer + Predictor | 显式文本摘要 | 无 |
| **TimeCAP** | Contextualizer + Predictor | 显式文本摘要 | 输入 + 提示增强 |

TimeCAP 与 [[event-driven-reasoning|事件驱动推理]]（VoT, ICLR 2026）在利用 LLM 理解上下文方面有相似动机，但 TimeCAP 使用 LLM 内生生成上下文（从数值推理），而 VoT 依赖外生文本（新闻等）。两者互补[^src-timecap]。

## 相关页面

- [[event-driven-reasoning]] — 事件驱动推理概念（VoT 范式）
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测任务
- [[vot]] — VoT 模型，利用外生文本的事件驱动 TS 预测
- [[aurora]] — Aurora 多模态生成式 TS 基础模型
- [[tats]] — TaTS 即插即用多模态框架
- [[timesfm]] — TimesFM decoder-only TS 基础模型
- [[chronos]] — Chronos 预训练 TS 语言模型
- [[autoformer]] — Autoformer 分解 Transformer（TimeCAP 的 baseline 之一）
- [[patchtst]] — PatchTST（TimeCAP 的 baseline 及 in-context sampler 对照）
- [[itransformer]] — iTransformer（TimeCAP 的 baseline 之一）

[^src-timecap]: [[source-timecap]]
