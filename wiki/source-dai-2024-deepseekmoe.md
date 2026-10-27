---
title: "Source: DeepSeekMoE 技术报告 — 细粒度专家分割与共享专家隔离 (arXiv:2401.06066)"
type: source-summary
tags:
  - source
  - mixture-of-experts
  - sparse-gating
  - expert-specialization
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: DeepSeekMoE（arXiv:2401.06066）

Damai Dai 等（DeepSeek-AI，联合北大、清华、南大），2024-01-11 的 v1 预印本，33 页；ACL 2024 著录来自用户，PDF 内未核实。本地为 `raw/deepseekmoe-2401-06066.pdf`，文本经 `pdftotext -layout` 转换。

## 核心论点

论文把稀疏 MoE 的不足归为两类：专家数有限（8 或 16）导致单个专家包含杂糅知识（knowledge hybridity），不同专家又重复学习公共知识（knowledge redundancy），二者共同压低专家特化[^src-dai-2024-deepseekmoe]。对策一是把每个 FFN 专家沿中间维切成 $m$ 份，总专家数 $mN$、激活 $mK$，在参数与计算量同 top-$K$ 持平下，激活组合数从 $\binom{N}{K}$ 变为 $\binom{mN}{mK}$（$N=16,K=2,m=4$ 时 120 → 4,426,165,368）[^src-dai-2024-deepseekmoe]。对策二是隔离 $K_s$ 个恒激活的共享专家，路由侧激活数降为 $mK-K_s$；式 (9) 中共享专家直接相加，不计门控权重[^src-dai-2024-deepseekmoe]。负载均衡采用 expert-level $f_iP_i$ 与 device-level $f_i'P_i'$ 两个辅助损失。门控固定为对专家质心的 softmax，无噪声、无 bias[^src-dai-2024-deepseekmoe]。

## 证据

2B 验证模型（1 共享 + 63 路由，激活 1+7，100B token）在总参数、激活参数与 4.3T FLOPs/2K token 全部对齐下，Pile loss 1.808 对 GShard 1.867，其余 10 项下游指标也全部最高（表 1）；并与专家参数 1.5 倍的 GShard×1.5（2.83B、5.8T）同为 1.808（表 2）[^src-dai-2024-deepseekmoe]。16B（16.4B / 2.8B）在 2T token 上以 40.5% 计算量与 DeepSeek 7B 相当（Pile BPB 0.74 对 0.75），以 39.6% 计算量在多数基准超过 LLaMA2 7B（HumanEval 26.8 对 14.6，MBPP 39.2 对 21.8；表 3、表 4）[^src-dai-2024-deepseekmoe]。145B 仅训 245B token，Pile loss 1.876 对 GShard 137B 1.961 与 DeepSeek 67B 1.905，计算量为后者 28.5%（表 6）[^src-dai-2024-deepseekmoe]。

## 局限

选择题任务落后于 DeepSeek 7B（MMLU 45.0 对 48.2，CEval 40.6 对 45.0），论文归因于 attention 参数仅约 0.5B；16B 不再采用更细粒度，以防专家过小降低效率；145B 为初步结果，总参数比 GShard 137B 多 6%。门控变体、balance factor 与设备分组数均无消融，图 3–6 仅有趋势，正文未列数值。机制与数据流见 [[deepseekmoe]]。

[^src-dai-2024-deepseekmoe]: [[source-dai-2024-deepseekmoe]]
