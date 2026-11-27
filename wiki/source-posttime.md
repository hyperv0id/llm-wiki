---
title: "PostTime: Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - multimodal-time-series
  - post-hoc-revision
  - sft
  - grpo
  - timesfm
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# PostTime: Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting

数值基础模型只看历史数值，无法接收以文本记录的外部冲击；通用大语言模型读得懂文本，却不擅长连续数值外推。PostTime 不让小语言模型从零外推，而是将其训练为预测后修订器：以冻结的 [[timesfm|TimesFM-2.5]] 预测为锚点，结合文本判断应修正、维持，还是判定文本无用而退回先验。核心难点不在「如何改」，而在「何时不该改」，作者把保守回退直接写进后训练配方；作者报告该配方优于冻结先验与零样本修订基线[^src-posttime]。

## 核心做法

- **冻结数值锚点**：连续外推始终由冻结的 TimesFM-2.5 完成，语言模型只输出对先验的文本修订[^src-posttime]。
- **仅预测期采样与保守回退**：训练数据构造时隐藏真实未来值，由外部教师大模型（Gemini）为每条样本采样多条推理轨迹，仅保留误差低于先验的较优轨迹；若采样全部不如先验，不丢弃该样本，而是插入「维持先验」的显式回退样本[^src-posttime]。
- **两阶段后训练**：先做有监督微调（LoRA SFT），再接群组相对策略优化（GRPO）；奖励以各样本自身先验误差为基准做截断线性校准，误差达到先验两倍即归零，以抑制极端偏离[^src-posttime]。

## 证据

1. **定位与主结论**：在防泄露刷新版 [[timesx|TimesX]] 多模态基准上总体优于冻结先验与免训练修订、集成基线（作者口径），属配方改进而非新算子[^src-posttime]。
2. **真实涨点归因**：主要增益来自第一阶段配方（先验锚定、仅预测期筛选与零有效回退），第二阶段 GRPO 精度增量微弱、主要稳定训练动力学[^src-posttime]。
3. **证据缺口**：分布外测试仅含 11 个变量且与强零样本集成同级；实验仅测试强数值先验，未验证弱先验场景[^src-posttime]。

## 相关页面

- [[timesx|TimesX]] — 复现对齐：沿用其多模态预测后修订基准与免训练基线，将推理期提示改为离线后训练。
- [[post-hoc-forecast-revision|预测后修订]] — 同槽位互斥：同为预测后修订，此处以冻结先验加保守回退替代数值检索修订。
- [[timesfm|TimesFM-2.5]] — 跨槽位正交：前者承担连续数值外推，本文只占文本预测后修订槽位。
- [[grpo-for-forecasting|GRPO 预测]] — 同槽位互斥：同为 GRPO 阶段，此处采用先验误差实例校准截断奖励，对方用固定尺度映射。

出版信息：NeurIPS 2026；Haoxin Liu、Yichen Zhou、Rajat Sen、B. Aditya Prakash（Georgia Tech / Google Research）。

[^src-posttime]: [[source-posttime]]
