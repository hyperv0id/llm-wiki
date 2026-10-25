---
title: "Latent Reasoning（隐空间推理）"
type: concept
tags:
  - latent-reasoning
  - test-time-compute
  - chain-of-thought
  - interpretability
  - monitorability
created: 2026-10-25
last_updated: 2026-10-25
source_count: 2
confidence: medium
status: active
---

# Latent Reasoning

Latent reasoning 指模型在**连续隐状态**里完成推理计算，不把中间步骤投影成可读的 token。它与 verbalized Chain-of-Thought 构成 test-time compute 的两条不同路线：CoT 用多生成 token 换取串行深度，代价是每一步都必须落到单个词表项上；latent reasoning 用 [[recurrent-depth|循环深度]]在同一 token 内部迭代，中间状态是高维向量，读者看不到[^src-2502-05171-recurrent-depth]。

## 两条轴的差别

论文把"内部推理必须投影成一个 verbalized next token"称为浪费，并把它与参数缩放、verbalized 推理缩放并列为互不冲突的三条轴（§1）[^src-2502-05171-recurrent-depth]。作者的表述是 latent 状态可以"同时深挖多个方向，而不是线性思考"[^src-2502-05171-recurrent-depth]。

代价也在同一处：进入隐空间后，中间计算失去文字形式。论文附录 A 自陈这"相比可读的思维链带来模型监督上的代价"，只提供了轨迹可视化的初步结果作为缓解[^src-2502-05171-recurrent-depth]。

## 机制证据（Huginn）

同一组权重、同一份数据下只改循环次数，可以直接测出 latent 迭代承担了多少计算[^src-2502-05171-recurrent-depth]：

- 只过 core 一次（$r=1$）时，0.8T checkpoint 的 GSM8K CoT 为 0.00、ARC-C 24.06、HellaSwag 29.34；$r=32$ 时分别到 34.80/42.08、38.23、65.21（Table 4）[^src-2502-05171-recurrent-depth]。
- 显式 CoT 的增益依赖循环打底：同一 checkpoint 加系统提示后，GSM8K 从 24.87/38.13（直接作答）到 34.80/42.08（CoT）；而 $r=1$ 时 CoT 完全不起作用（0.00）[^src-2502-05171-recurrent-depth]。
- 180B 早期 checkpoint 上，循环模型 GSM8K CoT 9.02 对同设置固定深度基线 1.82，差距来自架构而非训练量[^src-2502-05171-recurrent-depth]。

## 表征层面的观察

论文 §7 用三组分析说明 latent 迭代确实在"算"，而不只是重复同一变换；这些分析全部是定性的（无探针分类器、无注意力图、无统计检验）[^src-2502-05171-recurrent-depth]：

| 观察 | 内容 |
|---|---|
| 收敛速度按 token 变化 | 画每 token 每轮的 $\lVert s_i-s^{*}\rVert$（$s^{*}$ 取 $r=128$），关键 token 与回答开头收敛更慢，个别 token 出现振荡，距离非单调下降（Fig 11）[^src-2502-05171-recurrent-depth] |
| 轨迹几何三类 | 多数 token 收敛到不动点；数学/思考类 token 绕轨道（orbit）旋转，trivia 类不出现；少数 token 呈单向漂移（slider）（Fig 12、附录 B）[^src-2502-05171-recurrent-depth] |
| path independence | 从不同 $s_0$ 出发的轨迹收敛到同一不动点或同一轨道，是循环可外推到训练未见 $r$ 的直接依据[^src-2502-05171-recurrent-depth] |

作者对 orbit 与 slider 的解释是推测措辞（"可能服务周期性计算"、"也许可以用来数迭代次数"），不是实验结论[^src-2502-05171-recurrent-depth]。

逐 token 早停的行为也提供了侧面证据：以 $\mathrm{KL}(s_i,s_{i-1})<5\times10^{-4}$ 为退出判据时，MMLU 前 50 题里高中数学平均 12.7 步、道德情境 16.2 步，数学类最早退出（Fig 10）；warm start 用上一 token 的终态可把步数再降 1–2 步[^src-2502-05171-recurrent-depth]。

## 边界

- latent 迭代没有消除对数据的需求：$r=32$ 的 Huginn 在 ARC-E/MMLU 上是 69.91/31.38，OLMo-2-1124-7B 用 4T 精选 token 做到 82.79/60.56[^src-2502-05171-recurrent-depth]。
- 收益集中在串行推理成为瓶颈的任务上：$r$ 从 16 到 32 时 ARC-E 只动 0.42 分，SciQ 反而从 93.90 降到 93.50[^src-2502-05171-recurrent-depth]。
- 多操作数算术是明确的失败面：4–6 个操作数的单位数加法几乎做不对（4 operands 且 digit=1 时 0.3），且结果随循环次数顺序不一致（附录 A.1）[^src-2502-05171-recurrent-depth]。
- 论文与 Coconut 一类的连续思维的区别是：从头预训练带循环的模型，而不是微调已有固定深度模型，也不需要 CoT 数据（§6.3）[^src-2502-05171-recurrent-depth]。

## 可监督性

论文的担忧与外部政策讨论指向同一处：推理不进文字，监督窗口就少一条。二手报道引用的论证链是——在标准 Transformer 里"思维链是信息从后层流向前层的唯一途径"，循环给这条信息流增加了不出现在文本里的第二条通路；2025-07 的 CoT monitorability 立场论文在 "Novel architectures" 一节点名了 Coconut 与循环深度这一线工作，认为这类模型"可能不需要 verbalize 任何想法，从而失去 CoT 带来的安全优势"[^src-tosea-looped-transformer]。该立场论文的原文本页未独立核实，属二手转引[^src-tosea-looped-transformer]。

二手报道还声称 OpenAI 的 Astra 使用 recurrent depth（The Information 报道，公司既未确认也未否认），并记录其首席科学家以"当前前沿模型（含 Astra）的计算图深度在 GPT-4 的两倍以内"回应；报道自己把这个界归为"已确认但测量未公开"[^src-tosea-looped-transformer]。这两条都只有单一二手来源，不构成本页对任何厂商架构的事实断言[^src-tosea-looped-transformer]。

## 相关页面

- [[recurrent-depth]] — 承载 latent 迭代的架构本身
- [[test-time-computing-st]] — 时空预测里的 inference-time 计算，与本文的 LLM 侧 test-time compute 不是同一问题设定

[^src-2502-05171-recurrent-depth]: [[source-2502-05171-recurrent-depth]]
[^src-tosea-looped-transformer]: [[source-tosea-looped-transformer]]
