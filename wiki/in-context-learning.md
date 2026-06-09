---
title: "In-Context Learning (ICL)"
type: concept
tags:
  - in-context-learning
  - foundation-model
  - few-shot
  - prompting
created: 2026-06-09
last_updated: 2026-06-09
source_count: 1
confidence: medium
status: active
---

# In-Context Learning (ICL)

**In-context learning (ICL)** 指基础模型在**不更新权重**的前提下，仅凭推理时上下文中提供的少量演示（input–output 示例对）即时适配新任务的能力[^src-weathergfm]。任务由 prompt 上下文而非梯度下降指定，因此同一个冻结模型可被"提示"去完成多种未见任务。

## 在时空 / 天气基础模型中的应用

[[weathergfm|WeatherGFM]] 将 ICL 从文本推广到**视觉 in-context learning**：把"输入帧 → 目标帧"的演示对与查询拼成图像序列输入统一 ViT 主干，通过[[mixed-modal-masked-image-modeling|混合模态掩码图像重建]]一次性求解 10+ 种天气理解与预测任务，并靠[[weather-prompt|视觉提示]]（而非任务专属网络头）指定任务，从而在推理时泛化到未见任务与未见数据集[^src-weathergfm]。

## 与相关范式的关系

- [[weather-prompt]] — WeatherGFM 用以指定 ICL 任务的三类视觉提示格式
- [[mixed-modal-masked-image-modeling]] — 支撑视觉 ICL 的统一掩码重建目标
- [[historical-in-context-learning]] — 检索历史范例作为 in-context 指导的具体技术变体
- [[model-reprogramming]] — 另一类无需微调即复用冻结模型的范式（输入变换 + 输出投影）

[^src-weathergfm]: [[source-weathergfm]]
