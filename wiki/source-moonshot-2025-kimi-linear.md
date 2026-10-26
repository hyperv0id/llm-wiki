---
title: "Kimi Linear: An Expressive, Efficient Attention Architecture"
type: source-summary
tags:
  - llm
  - linear-attention
  - delta-rule
  - hybrid-attention
  - long-context
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Kimi Linear: An Expressive, Efficient Attention Architecture

Moonshot AI 技术报告（arXiv:2510.26692v2，2025-10-30 提交；PDF `raw/kimi-linear-2510-26692.pdf`，28 页）。核心主张：Kimi Delta Attention（KDA）+ 3:1 KDA/MLA 混合的 Kimi Linear 架构，在与 full attention 严格同配方对照下首次在短上下文、长上下文与 RL scaling 三个 regime 全面占优，同时 KV cache 最多省 75%、1M 上下文解码吞吐至多 6.3×[^src-moonshot-2025-kimi-linear]。

**机制**。KDA 把 GDN 的每头标量遗忘门细化为逐通道对角门 $\operatorname{Diag}(\alpha_t)$，等价于 DPLR 转移的 $a=b=k$ 特化，二级分块计算 4→2、再省 3 个矩阵乘，kernel 约比通用 DPLR 快 2×（§3、§6.2）[^src-moonshot-2025-kimi-linear]。混合 3 KDA + 1 Full MLA 层，MLA 全部 NoPE，位置信息由 KDA 的门控衰减隐式承载（§4、§6.1）[^src-moonshot-2025-kimi-linear]。48B/3B 激活 MoE，对照实验统一 1.4T token，发布 checkpoint 5.7T、上下文 1M（§5.4）[^src-moonshot-2025-kimi-linear]。

**证据**。合成任务上 KDA 全面高于 GDN，Mamba2 全败（Figure 4）；3:1 比例验证 PPL 5.65 最低（Table 1）；scaling law 约 1.16× 计算效率（Figure 5）；1.4T 下短上下文（Tables 3–4）、长上下文（Table 5 平均 54.5 对 MLA 52.2）、数学 RL（Figure 6）三线领先，个别基准（EvalPlus、MATH500、LongBench V2、Frames）例外；5.7T instruct 版 RULER@1M 94.8（Tables 8–9）[^src-moonshot-2025-kimi-linear]。

**边界**。纯线性部分仍受有限状态约束，以 3:1 混入全局注意力维持检索；scaling law 沿用 MLA 超参未细调；与 Moonlight 的长上下文对照因对方 8K 上限不完整（§5.3、§7.1、附录 D）[^src-moonshot-2025-kimi-linear]。

机制详解见 [[kimi-delta-attention]] 与 [[gated-delta-rule]]。

[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
