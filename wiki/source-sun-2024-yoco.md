---
title: "You Only Cache Once: Decoder-Decoder Architectures for Language Models"
type: source-summary
tags:
  - llm
  - kv-cache
  - architecture
  - efficient-inference
  - long-context
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# You Only Cache Once（YOCO）

Sun et al.（Microsoft Research 与清华大学，arXiv:2405.05254v2，2024-05；PDF `raw/yoco-2405-05254.pdf`）针对长上下文推理的两个瓶颈：KV cache 随层数×长度增长，65B 模型（GQA + 8-bit KV）在 512K 上下文占约 86 GB，超过单张 H100-80GB；prefill 延迟随长度平方增长，7B 模型加 Flash-Decoding 后四卡填 450K 上下文仍需约 110 s（§1）[^src-sun-2024-yoco]。

机制：模型堆叠 $L$ 个 block，前 $L/2$ 层 self-decoder 用常数推理内存的高效自注意力（gated retention 或滑窗注意力）编码出唯一的全局 KV cache，后 $L/2$ 层 cross-decoder 以标准 cross-attention 读取同一份 cache，外部行为等同 decoder-only Transformer（§2）；cache 记账从 $O(LND)$ 降至 $O((N+L)D)$（Table 1），prefill 可在进入 cross-decoder 前提前退出，注意力项从 $O(LN^2D)$ 降至 $O(LND)$（Table 2）[^src-sun-2024-yoco]。

证据：3B 模型按 StableLM-3B-4E1T 配方训练 1.6T token，Eval Harness 平均 0.636、1M 扩展版 0.645，与对照 Transformer 相当（Table 3）；160M–13B scaling 曲线与 Llama 式 Transformer 相当，gRet 版损失最低（Figure 4、Table 8）；上下文扩至 1M 后 NIAH 近满分（Figure 5），128K 多针检索与 7B 长上下文模型可比（Table 4）；部署上 512K prefill 从 180 s 降至 6 s 内，吞吐从 4.5 升至 43.1 token/s，65B 的 KV cache 约省 $80\times$（§4.4、Figures 7–10）[^src-sun-2024-yoco]。

局限："once" 仅指全局 cache，self-decoder 仍需常数窗口或 $d\times d$ 状态（脚注 2）；全局记忆只有一份，由 self-decoder 末端表示 $X^{L/2}$ 一次投影而来，cross 层只读；质量对照限于 3B（1.6T token）与 160M–13B（各 10B token）两个量级，无指令微调与对话评测（§4）[^src-sun-2024-yoco]。

## 相关页面

[[yoco-architecture]] 涵盖结构、数据流、cache 记账与全部实验数字；[[native-sparse-attention]]、[[compressed-sparse-attention]]、[[multi-head-latent-attention]]、[[multi-query-attention]]、[[paged-attention]] 为其他「省 KV」轴的对照；[[kv-cache-compression]] 是各轴的汇总页。

[^src-sun-2024-yoco]: [[source-sun-2024-yoco]]
