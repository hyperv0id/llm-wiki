---
title: "Overcoming the Modality Gap in Context-Aided Forecasting (CAF-7M & DoubleCast)"
type: source-summary
tags:
  - multimodal-time-series
  - text-fusion
  - chronos
  - data-curation
  - icml-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# Overcoming the Modality Gap in Context-Aided Forecasting

- **来源**：Zheng\*、Marcotte\*、Ashok、Williams、Sun、Drouin、Zantedeschi（ServiceNow AI Research / McGill / DIRO-UdeM / Mila），ICML 2026（PMLR 306，Seoul）[^src-doublecast]。
- **核心问题与数据建构**：针对多模态时序预测常弱于纯数值单模态的失效现象，作者提出根因不在融合架构，而在现有语料中的文本从未被经验验证对预测有用[^src-doublecast]。**CAF-7M** 采用「先生成再检验」流程：先由 Llama-3.3-70B 读取完整历史与未来窗口反推 Background/Scenario/Constraint 解释文本，再由强上下文模型在有/无文本下分别预测，仅保留文本使 CRPS 明显下降的窗口作为经验证测试集（覆盖 $65$ 个真实数据集/$11$ 域，$743$ 万训练窗与 $904$ 条验证测试窗，按 $\mathrm{MASE}>1.5$ 分 HARD $490$ / EASY $414$）[^src-doublecast]。
- **去黑话配方（DoubleCast）**：以冻结的 [[chronos|Chronos]]（T5-large）分词器与编码器为数值主干，冻结 Qwen3-14B 取倒数第二层词元级文本隐状态；在 Chronos 解码器各块内按序执行掩码自注意力、时序编解码交叉注意力、新增 `DualT5` 文本交叉注意力适配器与前馈网络，经离散分位数头输出分布[^src-doublecast]。因缺乏文本内细粒度时间索引绑定，局部时段信号易被平滑[^src-doublecast]。

## 证据

1. **主结论**：CAF-7M 零样本优于 Chronos/Chronos-2，CGTSF 多数领先 ChatTime，GIFT-Eval 保原精度[^src-doublecast]。
2. **归因**：无训练架构消融；推理去文本/错配使 CRPS 恶化 $18.3\%/26.5\%$，Time-LLM 反降 $8.0\%$[^src-doublecast]。
3. **缺口**：MSPG（$H=96>64$）与 [[time-mmd|Time-MMD]]（$7/9$ 域）存限；训练窗未过滤；GPT-5.2 存在自证偏置[^src-doublecast]。

## 相关页面

- [[doublecast]] · [[multimodal-time-series-forecasting]] · [[constrained-text-fusion]] · [[time-mmd]] · [[chronos]] · [[aurora]] · [[circular-dependency-in-multimodal-fusion]]

[^src-doublecast]: [[source-doublecast]]
