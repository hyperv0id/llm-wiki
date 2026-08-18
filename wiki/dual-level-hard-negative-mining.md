---
title: "Dual-level Hard Negative Mining"
type: technique
tags:
  - contrastive-learning
  - retrieval
  - cross-modal
  - time-series
  - hard-negative
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Dual-level Hard Negative Mining

Dual-level Hard Negative Mining 是 [[trace|TRACE]] 在跨模态对齐阶段（Stage 2）提出的对比学习策略，在 sample-level 和 channel-level 两个粒度上动态挖掘硬负样本[^src-trace-neurips2025]。

## 两个级别

### Sample-level

对齐 [CLS] 嵌入 h_CLS 与样本级上下文文本嵌入 z_cxt。硬负样本从 batch 内其他样本中按 TopK 相似度选取[^src-trace-neurips2025]：

- N_cxt^(i) = TopK sim(h_CLS^(i), z_cxt^(j)) | j ≠ i
- N_cxt^(i,text) = TopK sim(z_cxt^(i), h_CLS^(j)) | j ≠ i

### Channel-level

对齐 [[channel-identity-token|CIT]] 嵌入 h_c 与通道级文本嵌入 z_c。硬负样本包括两类 distractor[^src-trace-neurips2025]：

- **Intra-instance negatives**：同一实例内其他通道的文本嵌入 z_c' (c' ≠ c)
- **Inter-instance negatives**：不同实例同索引通道的文本嵌入 z_c^(j) (j ≠ i)

这使模型能区分"看起来相似但语义不同"的通道模式和跨实例同通道但语义发散的描述[^src-trace-neurips2025]。

## 损失

双向 InfoNCE，sample-level 和 channel-level 各两个方向（text→ts, ts→text），总损失 L_align = (L_global^text→ts + L_global^ts→text + λ_ch · (L_channel^text→ts + L_channel^ts→text)) / 2，λ_ch=1.0[^src-trace-neurips2025]。

## 消融

- 移除 channel-level 对齐：检索精度一致下降[^src-trace-neurips2025]。
- 移除 cross-attention 模块：在小 K（少量负样本）时精度显著下降[^src-trace-neurips2025]。
- 文本编码器质量直接影响硬负区分能力：nomic > bge > MiniLM[^src-trace-neurips2025]。

## 与现有对比学习的区别

传统 [[contrastive-learning|对比学习]]（如 CLIP）仅做 sample-level 随机负采样。文本描述常引用具体变量（如温度峰值、风速），单全局嵌入无法精确对齐。TRACE 的 channel-level 对齐显式建模单通道时序信号与对应文本的交互，提升语义精度并促进表示模块化[^src-trace-neurips2025]。

## 相关

- [[trace]] — TRACE 模型
- [[contrastive-learning]] — 对比学习
- [[channel-identity-token]] — CIT 嵌入
- [[cross-modal-misalignment]] — 跨模态对齐的理论限制

[^src-trace-neurips2025]: [[source-trace-neurips2025]]
