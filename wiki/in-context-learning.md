---
title: "In-Context Learning (ICL)"
type: concept
tags:
  - in-context-learning
  - foundation-model
  - few-shot
  - prompting
created: 2026-06-09
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# In-Context Learning (ICL)

**In-context learning (ICL)** 指基础模型在**不更新权重**的前提下，仅凭推理时上下文中提供的少量演示（input–output 示例对）即时适配新任务的能力[^src-weathergfm]。任务由 prompt 上下文而非梯度下降指定，因此同一个冻结模型可被"提示"去完成多种未见任务。

## 在时空 / 天气基础模型中的应用

[[weathergfm|WeatherGFM]] 将 ICL 从文本推广到**视觉 in-context learning**：把"输入帧 → 目标帧"的演示对与查询拼成图像序列输入统一 ViT 主干，通过[[mixed-modal-masked-image-modeling|混合模态掩码图像重建]]一次性求解 10+ 种天气理解与预测任务，并靠[[weather-prompt|视觉提示]]（而非任务专属网络头）指定任务，从而在推理时泛化到未见任务与未见数据集[^src-weathergfm]。

## 同题跨遍反馈与少样本演示

[[trace-as-state|Trace as State]] 的已评估实现同样冻结模型，但提供给第二遍的文本来自模型对**同一道题**的首轮推理，而不是其他题目的 input–output 演示。作者把这些可能有错的轨迹当作 textual state proxy，比较 `[T,x,q]` 与 `[x,T,q]`；单独返回的首轮可见答案不放入 `T`（§3、Appendix C）。[^src-trace-as-state]

[INFERENCE] 这里适合沿“推理时用上下文携带信息”的维度建立联系，而不宜把所有文本反馈都称作 few-shot demonstration。是否需要训练、文本来自哪里、文本何时能参与原文表示，是三个应分别说明的设计选择。[^src-weathergfm][^src-trace-as-state]

## 与相关范式的关系

- [[weather-prompt]] — WeatherGFM 用以指定 ICL 任务的三类视觉提示格式
- [[mixed-modal-masked-image-modeling]] — 支撑视觉 ICL 的统一掩码重建目标
- [[historical-in-context-learning]] — 检索历史范例作为 in-context 指导的具体技术变体
- [[model-reprogramming]] — 另一类无需微调即复用冻结模型的范式（输入变换 + 输出投影）
- [[test-time-policy-optimization]] — 同样免权重更新的另一条路线：以奖励信号直接调制输出分布，而非把经验放回 prompt（[[jitrl|JitRL]]）
- [[trace-as-state]] — 同题轨迹前置后重读
- [[trace-as-state-positioning]] — 重读、轨迹状态与递归证据回放的比较

[^src-weathergfm]: [[source-weathergfm]]
[^src-trace-as-state]: [[source-trace-as-state]]
