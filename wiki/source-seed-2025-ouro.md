---
title: "Scaling Latent Reasoning via Looped Language Models（Ouro）"
type: source-summary
tags:
  - latent-reasoning
  - recurrent-depth
  - weight-tying
  - test-time-compute
  - language-model
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Scaling Latent Reasoning via Looped Language Models（Ouro）

Rui-Jie Zhu、Zixuan Wang、Kai Hua 等（ByteDance Seed、UC Santa Cruz、Princeton、Mila 等），arXiv:2510.25741v6（cs.CL，2026-10-11），54 页；PDF 归档 `raw/ouro-2510-25741.pdf`[^src-seed-2025-ouro]。

## 问题

LLM 扩容长期依赖参数、数据与推理期 CoT token；论文探讨权重共享循环栈能否在万亿 token 级提供第三条轴，即固定参数预算内用隐状态迭代加深计算[^src-seed-2025-ouro]。

## 方法

Ouro 将同一组 24/48 层 Transformer 在单个 token 隐状态上迭代 4 轮，参数不随轮数增加。每个循环步均接 LM head 与退出门，训练目标按退出分布加权各步 NLL，并在均匀先验下加熵正则（$\beta$ 从 0.1 降至 0.05）。第二阶段冻结 LM，用相邻轮损失下降量训练门；推理时按累积退出概率的 $q$ 分位早停[^src-seed-2025-ouro]。1.4B/2.6B 共用四段数据，合计 7.7T token[^src-seed-2025-ouro]。

## 证据

论文报告 1.4B 在多数基准上对齐 4B dense，2.6B 对齐 8B：1.4B 的 BBH 71.02、GSM8K 78.92、MATH500 82.40；2.6B 的 MMLU-Pro 55.73、BBH 80.46、MATH500 90.85（Table 7–8）[^src-seed-2025-ouro]。Thinking 版在 AIME24/25、OlympiadBench、BeyondAIME 上与 Qwen3-4B/8B 互有胜负（Table 9）[^src-seed-2025-ouro]。增益归于知识操作而非容量：Capo 上知识容量约 2 bits/parameter 且不随循环变化，Mano 上同参数对照更准，3-hop QA 更省样本，MMLU 推理类子项增益远大于事实类（Table 15）；图可达性构造给出 $O(1+\log D)$ 轮（Theorem 1）[^src-seed-2025-ouro]。解码期只留末轮 KV 可将缓存降至 $1/4$，GSM8K 为 78.85 对 78.92（Table 14）；Thinking 版在 HEx-PHI 上的 harmful rate 为 0.009（1.4B）与 0.003（2.6B），Qwen3-4B-Thinking 为 0.009（§7.1）[^src-seed-2025-ouro]。

## 局限

仅在 $T=4$ 训练，$T=5$–$8$ 外推普遍退化；同参数小规模对照中标准 Transformer 仍高 0.015–0.039（Table 18）；主实验缺少同 token 与同算力 dense 对照；RLVR 未超过 SFT；prefill 无法复用 KV；忠实性分析为无干预的观察性代理；定理假设偏窄且每轮 $O(n^3)$ 工作[^src-seed-2025-ouro]。

[^src-seed-2025-ouro]: [[source-seed-2025-ouro]]
