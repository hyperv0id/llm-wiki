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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# Overcoming the Modality Gap in Context-Aided Forecasting

多模态时序预测反复出现融合文本后反而不及纯数值单模态的现象[^src-doublecast]。论文诊断根因多半不在融合架构，而在现有数据集的文本从未被验证是否真含超越数值历史的预测信息[^src-doublecast]。解法是先生成、再验证：大模型对照历史与未来窗口反推解释文本，用有/无文本预测对比筛掉无增益窗口，得到 700 万窗口的 CAF-7M 语料；配套的 DoubleCast 在冻结 Chronos 解码块中插入文本交叉注意力，证明这类文本可被真正吸收并迁移到真实数据[^src-doublecast]。

## 核心做法

**数据侧（CAF-7M）**：Llama-3.3-70B 对照完整历史与未来窗口反推 Background/Scenario/Constraint 解释文本；强预测模型再做有/无文本两次预测，仅保留文本明显降低 CRPS 的窗口进入验证测试集（904 条，分 HARD/EASY），覆盖 65 个数据集、11 个领域；约 743 万训练窗未筛[^src-doublecast]。

**模型侧（DoubleCast）**：复用冻结的 [[chronos|Chronos]]（T5-large）分词器与编码器作数值主干，文本侧取冻结 Qwen3-14B 倒数第二层词元隐状态；每个解码块内在掩码自注意力与时序交叉注意力之后串行插入可训练的文本交叉注意力适配器，经离散分位数头输出分布[^src-doublecast]。

## 证据

1. **定位与主结论**：CAF-7M 训练的 DoubleCast 在验证集上仅次于筛选器自身 GPT-5.2；零样本迁移到真实文本基准 CGTSF 多数优于 Chronos 与 ChatTime，GIFT-Eval 保持原精度[^src-doublecast]。
2. **真实涨点归因**：无架构消融；涨点归数据筛选，去文本 CRPS 恶化 $18.3\%$、错配 $26.5\%$，Time-LLM 反改善 $8.0\%$[^src-doublecast]。
3. **证据缺口与负结果**：超长视界基准 MSPG 上超训练上限后零样本落后；Time-MMD 上 7/9 域胜出；GPT-5.2 自筛自评存在偏置[^src-doublecast]。

## 局限

论文自述：筛选偏向验证器可利用的模式，文本多样性受限；上下文由未来窗反推，较部署真实文本更干净，存在生态效度差距；文本时间引用缺细粒度绑定，局部信号可能被弥散[^src-doublecast]。

## 相关页面

- 同构/特例：[[doublecast]] — 本方法的机制与数据流页
- 同槽位互斥：[[constrained-text-fusion|CFA]]、[[time-llm|Time-LLM]] — 同为多模态时序融合方案：CFA 在融合层用低秩瓶颈限容、Time-LLM 在输入端前缀拼接，本文改为数据侧效用筛选加解码块内注入
- 瓶颈接力 $\Delta$：[[time-mmd|Time-MMD]] — 在对齐与防泄漏拆分之后补上「文本是否对预测有益」的效用验证门槛
- 复现对齐：[[chronos|Chronos]] — 冻结其分词器与编码器作数值骨干，GIFT-Eval 上复现原零样本精度

出处：Zheng\*、Marcotte\* 等（ServiceNow AI Research / McGill / DIRO-UdeM / Mila），ICML 2026。

[^src-doublecast]: [[source-doublecast]]
