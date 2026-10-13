---
title: "Source: MoBA: Mixture of Block Attention for Long-Context LLMs"
type: source-summary
tags:
  - source
  - attention
  - sparse-attention
  - mixture-of-experts
  - long-context
  - llm
created: 2026-10-08
last_updated: 2026-10-08
source_count: 1
confidence: medium
status: active
---

# Source: MoBA (arXiv:2502.13189)

- **作者**：Enzhe Lu, Zhejun Jiang, Jingyuan Liu 等 25 人，全部署名 Moonshot AI
- **时间**：2025-02-18（v1），15 页
- **链接**：https://arxiv.org/abs/2502.13189
- **本地**：`raw/moba-mixture-of-block-attention.md`
- **代码**：https://github.com/MoonshotAI/MoBA

## 核心论点

论文提出 Mixture of Block Attention（[[moba-attention]]）：把 MoE 的 top-k 门控应用于 attention——KV 上下文分块为"专家"，亲和分数取 query 与块内 key 均值的内积，每个 query 只对 top-k 个块做 attention；当前块强制选择并加 causal mask 以保持因果性（类比 shared expert）。无新增参数，top-k 全选时退化为 full attention，可在层间与训练中动态切换。设计原则 "less structure"：不用 sink/window 一类预定义稀疏偏置，也不做线性 attention 近似（转换成本高，复杂推理证据有限）。sliding window 与 attention sink 都是 MoBA 的特例（固定门控即可得到）。

## 贡献

1. **实现**：门控分配、query 重排、变长 FlashAttention、online softmax 合并的五步流程，无需自定义 CUDA kernel。
2. **实证**：568M–2.1B 缩放实验验证 loss 与 full attention 相差 1e-3 以内（trailing loss 有缺口，需 90/10 两阶段或逐层混合修复）；Llama 3.1 8B 持续预训练至 1M 上下文，RULER 128K 0.7818 vs full 0.7849；attention 层 1M prefill 加速 6.5 倍、10M 时 16 倍。
3. **部署**：已上线 Kimi 长上下文请求。

## 局限

评测中 MoBA 仅用于 prefill、生成期切回 full attention，生成期稀疏质量未单独评估；trailing loss 缺口依赖混合策略修复；部署无线上指标。
