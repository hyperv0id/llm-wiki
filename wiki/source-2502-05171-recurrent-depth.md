---
title: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"
type: source-summary
tags:
  - recurrent-depth
  - latent-reasoning
  - test-time-compute
  - architecture
  - language-model
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach

Jonas Geiping、Sean McLeish、Neel Jain、John Kirchenbauer、Siddharth Singh、Brian R. Bartoldson、Bhavya Kailkhura、Abhinav Bhatele、Tom Goldstein（ELLIS Institute Tübingen / MPI-IS / Tübingen AI Center、University of Maryland、Lawrence Livermore National Laboratory），arXiv:2502.05171v2（cs.LG，2025-02-17）[^src-2502-05171-recurrent-depth]。训练文本由 `pdftotext -layout` 导出（`raw/2502.05171-recurrent-depth.txt`）。

## 问题

推理模型扩测试时算力的通行做法是多生成 CoT token，代价是中间计算必须逐个投影成词表项、上下文随之变长。论文问：能否把串行计算放进连续隐空间，用**循环深度**（对一个共享权重块迭代若干次）在单个 token 内部换取深度，而不产生额外 token[^src-2502-05171-recurrent-depth]。作者把这条路线定位为与参数缩放、token 缩放并列的第三条轴（§1）[^src-2502-05171-recurrent-depth]。

## 方法

模型拆成 prelude（2 层，嵌入输入）、core（4 层共享权重块，迭代 $r$ 次）、coda（2 层 + tied 输出头）三段，主模型 3.5B 参数、$h=5280$、等效深度 $2+4r+2$[^src-2502-05171-recurrent-depth]。每轮把输入嵌入与上一轮状态拼接后经 $A:\mathbb{R}^{2h}\to\mathbb{R}^h$ 重新注入，初值 $s_0$ 从截断正态采样，两者共同保证迭代收敛到与初值无关的不动点（path independence）；迭代步号刻意不进模型，因为 step embedding 会破坏这一性质[^src-2502-05171-recurrent-depth]。训练时 $r$ 由 log-normal Poisson 采样（$\bar r=32$，$\sigma=1/2$），损失只在最后一轮算 next-token 交叉熵；反向只穿最后 $k=8$ 轮（深度方向截断 BPTT）并逐迭代 checkpoint，使显存与 $r$ 无关[^src-2502-05171-recurrent-depth]。推理期全部 zero-shot：扫 $r$、按相邻步 KL 逐 token 早停、KV 预算 $i\bmod s$、warm start、self-speculative decoding（§6）[^src-2502-05171-recurrent-depth]。

## 证据

Huginn-0125 训练 800B token（47000 步，Frontier 4096×MI250X，16M token/step），非循环基线 180B token[^src-2502-05171-recurrent-depth]。$r=32$ 时 ARC-E 69.91、ARC-C 38.23、HellaSwag 65.21、MMLU 31.38、GSM8K CoT 34.80/42.08（Table 1、2）[^src-2502-05171-recurrent-depth]。同设置的循环 vs 非循环对照（Table 4）是核心因果证据：180B 时 ARC-E 53.62 vs 46.42、GSM8K CoT 9.02 vs 1.82；而 $r=1$ 的 0.8T 模型 GSM8K CoT 仍为 0.00，说明增益全部由迭代产生、不在 prelude/coda[^src-2502-05171-recurrent-depth]。测试时最多外推到 $r=64$ 不退化，饱和点随任务变化（HellaSwag 约 8 次，ARC-C 的 25–50-shot 约 32 次）[^src-2502-05171-recurrent-depth]。作者给出的能力口径是 FLOP 等价：预训练算力接近 32B 参数 transformer，测试时接近 50B 固定深度 transformer；精度上大致与 OLMo-7B 相当、落后 OLMo-2（§5.1）[^src-2502-05171-recurrent-depth]。

## 局限

论文自述仍是 proof-of-concept、只跑了一个大模型、学习率从未 cooldown（§10）[^src-2502-05171-recurrent-depth]。架构消融基本是定性的，没有逐项数值表；显式 CoT 只在循环打底后才有效；4–6 个操作数的加法几乎做不对；代码落后专用模型；多段循环、与 MoE/线性注意力结合、按难度做 RL 都留作 future work[^src-2502-05171-recurrent-depth]。训练存在静默失败模式（学到忽略状态后 $r=1$ 与 $r=32$ 的验证困惑度相同），需要靠 val-ppl-vs-recurrence 诊断才发现[^src-2502-05171-recurrent-depth]。

[^src-2502-05171-recurrent-depth]: [[source-2502-05171-recurrent-depth]]
