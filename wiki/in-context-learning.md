---
title: "In-Context Learning (ICL)"
type: concept
tags:
  - in-context-learning
  - foundation-model
  - few-shot
  - prompting
created: 2026-06-09
last_updated: 2026-11-24
source_count: 4
confidence: medium
status: active
---

# In-Context Learning (ICL)

**In-context learning (ICL)** 指基础模型在**不更新权重**的前提下，仅凭推理时上下文中提供的少量演示（input–output 示例对）即时适配新任务的能力[^src-weathergfm]。任务由 prompt 上下文而非梯度下降指定，因此同一个冻结模型可被"提示"去完成多种未见任务。

## 在时空 / 天气基础模型中的应用

[[weathergfm|WeatherGFM]] 将 ICL 从文本推广到**视觉 in-context learning**：把"输入帧 → 目标帧"的演示对与查询拼成图像序列输入统一 ViT 主干，通过[[mixed-modal-masked-image-modeling|混合模态掩码图像重建]]一次性求解 10+ 种天气理解与预测任务，并靠[[weather-prompt|视觉提示]]（而非任务专属网络头）指定任务，从而在推理时泛化到未见任务与未见数据集[^src-weathergfm]。

## 同题跨遍反馈与少样本演示

[[trace-as-state|Trace as State]] 的已评估实现同样冻结模型，但提供给第二遍的文本来自模型对**同一道题**的首轮推理，而不是其他题目的 input–output 演示。作者把这些可能有错的轨迹当作 textual state proxy，比较 `[T,x,q]` 与 `[x,T,q]`；单独返回的首轮可见答案不放入 `T`[^src-trace-as-state]。

[INFERENCE] 这里适合沿“推理时用上下文携带信息”的维度建立联系，而不宜把所有文本反馈都称作 few-shot demonstration。是否需要训练、文本来自哪里、文本何时能参与原文表示，是三个应分别说明的设计选择。[^src-weathergfm][^src-trace-as-state]

## 时序上下文条件化与隐状态延续

在单变量时序预测语境下，[[tirex|TiRex]] 也使用「上下文学习（in-context learning）」一词，但其机制属于**同词异构用法**：不构造多条示例的输入–输出演示对，而是依靠纯 `sLSTM` 解码器追踪单条历史窗口中的循环状态，并在预训练中结合连续块掩码（CPM）及幅值调制、截断、周期尖峰三项增强，使模型在推理期只需凭单序列历史上下文与末尾缺失占位即可零样本泛化至长短不同预测窗[^src-tirex]。

## 预测后文本修订的示例与保守回退（跨槽位正交）

在多模态预测后修订（post-forecasting revision）槽位上，[[timesx|TimesX]] 的 `TextRev-ICL` 与上述直接条件化主干的用法呈**跨槽位正交**关系[^src-timesx]。免训练的直接文本修订代理（`TextRev`）容易放大预测方差与尾部误差，整体弱于简单的等权集成 `AvgEns`；`TextRev-ICL` 则在推理上下文中为大语言模型提供历史修订示例，并显式加入「保守不改」的回退选项，从而抑制过度修订并超越等权集成表现[^src-timesx]。

| 方法代表 | 作用槽位 | 上下文内容来源 | 参数更新 | 控制稳定性/防退化机制 |
|---|---|---|---|---|
| [[weathergfm\|WeatherGFM]] | 主干预测（视觉输入端） | 多题「输入帧–目标帧」演示对与视觉提示 | 推理冻结（需混合模态预训练） | 统一掩码重建目标[^src-weathergfm] |
| [[trace-as-state\|Trace as State]] | 同题第二遍推理输入端 | 同一道题的首轮推理轨迹文本 | 推理冻结 | 仅保留轨迹状态代理，隔离首轮答案[^src-trace-as-state] |
| [[tirex\|TiRex]] | 单序列预测（隐状态递推） | 单条历史观测窗口与末尾缺失占位 | 推理冻结（需 CPM 预训练） | `sLSTM` 递推状态连续延续[^src-tirex] |
| [[timesx\|TimesX (TextRev-ICL)]] | 预测后文本修订槽位 | 历史修订示例 + 四类真实时钟上下文 | 推理免训练 | 显式「保守不改」回退选项以抑制修订方差[^src-timesx] |

## 与相关范式的关系

- [[timesx]] — `TextRev-ICL` 在预测后文本修订槽位引入历史修订示例与保守回退（跨槽位正交）
- [[tirex]] — 以单序列历史窗口与连续块掩码驱动零样本预测（同词异构的时序上下文条件化）
- [[weather-prompt]] — WeatherGFM 用以指定 ICL 任务的三类视觉提示格式
- [[mixed-modal-masked-image-modeling]] — 支撑视觉 ICL 的统一掩码重建目标
- [[historical-in-context-learning]] — 检索历史范例作为 in-context 指导的具体技术变体
- [[model-reprogramming]] — 另一类无需微调即复用冻结模型的范式（输入变换 + 输出投影）
- [[test-time-policy-optimization]] — 同样免权重更新的另一条路线：以奖励信号直接调制输出分布，而非把经验放回 prompt（[[jitrl|JitRL]]）
- [[trace-as-state]] — 同题轨迹前置后重读
- [[trace-as-state-positioning]] — 重读、轨迹状态与递归证据回放的比较

[^src-weathergfm]: [[source-weathergfm]]
[^src-trace-as-state]: [[source-trace-as-state]]
[^src-tirex]: [[source-tirex]]
[^src-timesx]: [[source-timesx]]
