---
title: "MiniMax-01: Scaling Foundation Models with Lightning Attention"
type: source-summary
tags:
  - llm
  - lightning-attention
  - linear-attention
  - moe
  - long-context
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# MiniMax-01: Scaling Foundation Models with Lightning Attention

MiniMax 开源基础模型技术报告（arXiv:2501.08313，2025-01；PDF `raw/minimax-01-2501-08313.pdf`）的核心主张是，线性注意力首次在基础模型规模落地：456B 总参、45.9B 激活的 MoE，采用每 7 个 transnormer 块（lightning attention）配 1 个 softmax attention 块的混合架构，共 80 层；训练上下文扩至 1M token，推理外推至 4M[^src-minimax-2025-01]。

**问题与机制**。纯线性注意力省算力但检索能力弱。论文把 lightning attention 定位为 TransNormer 的 I/O-aware 分块实现：块内用左积（含因果掩码的 $B\times B$ 注意力）、块间用右积（$\mathbf{KV}$ 状态外积累加），复杂度 $O(nd^2+nBd)$（§2.2.1）[^src-minimax-2025-01]。针对 lightning attention "demonstrates limited retrieval capabilities"，softmax 层按每 8 层 1 个的间隔补足检索（§2.2.2、§2.2.3）[^src-minimax-2025-01]。

**证据**。70M–7B scaling law 显示：同预算下 hybrid 用更多参数与 token 但 loss 更低（Table 2）；1B/3B 对照中 hybrid-lightning 在 NIAH 上达 95.7/98.0，远超 hybrid-window 的 53.9/57.9（Table 3–4）；28B/5B MoE 消融中每 8 层的前 7 层换用 lightning，BBH 达 28.2→32.2（Table 5）。最终模型 C-SimpleQA 67.4 为全场最高，RULER 自 128k 起领先且 1M 得分 0.910（Table 8–9）[^src-minimax-2025-01]。基础设施侧：MoE 纯通信开销降 50%，H20 端到端推理 MFU 超 75%，1,024,000 token 处 lightning attention 延迟占比 <12% 而 softmax 注意力实现占 95%（§3）[^src-minimax-2025-01]。

**边界**。保留 1/8 softmax 是妥协，目标是未来完全去掉 softmax（§7）；长上下文评测集偏人工场景；预训练编程数据不足，MBPP+ 71.7 在对照表中垫底，HumanEval 86.9 仅高于 Gemini-1.5-Pro 与 Qwen2.5-72B 的 86.6；线性注意力九年间未被主流 LLM 采用，主因是 causal 场景右积需 cumsum、阻碍高效并行（§2.2）[^src-minimax-2025-01]。

机制详解见 [[lightning-attention]]。

[^src-minimax-2025-01]: [[source-minimax-2025-01]]
