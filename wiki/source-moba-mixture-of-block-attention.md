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
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: MoBA (arXiv:2502.13189)

作者为 Enzhe Lu、Zhejun Jiang、Jingyuan Liu 等 25 人，其中 23 人来自 Moonshot AI，另 2 人为 Mingxing Zhang（清华大学）与 Jiezhong Qiu（之江实验室/浙江大学）。论文于 2025-02-18 发布（v1，15 页），链接为 https://arxiv.org/abs/2502.13189，本地见 `raw/moba-mixture-of-block-attention.md`，代码开源于 https://github.com/MoonshotAI/MoBA。

## 核心论点

论文提出 Mixture of Block Attention（[[moba-attention]]），将 MoE 的 top-k 门控引入 attention：KV 上下文分块作为“专家”，亲和分数取 query 与块内 key 均值的内积，每个 query 仅对 top-k 个块计算 attention；当前块强制选入并加 causal mask 保持因果性（类似 shared expert）。方法不增加参数，top-k 全选时退化为 full attention，可在层间与训练中动态切换。设计遵循 "less structure" 原则，不使用 sink/window 等预定义稀疏偏置，也不做线性 attention 近似（转换成本高且复杂推理证据有限）。sliding window 与 attention sink 均为 MoBA 在固定门控下的特例。

## 贡献

1. **实现**：门控分配、query 重排、变长 FlashAttention、online softmax 合并的五步流程（当前块 causal 与历史块 non-causal 分开计算，结合 FlashAttention 与 MoE 两方优化技术）。
2. **实证**：568M–2.1B 缩放实验验证 loss 与 full attention 相差 1e-3 以内（trailing loss 有缺口，需 90/10 两阶段或逐层混合修复）；Llama 3.1 8B 持续预训练至 1M 上下文，RULER 128K 0.7818 vs full 0.7849；attention 层 1M prefill 加速 6.5 倍、10M 时 16 倍。
3. **部署**：已上线 Kimi 长上下文请求。

## 局限

评测中 MoBA 仅用于 prefill，生成期切回 full attention，未单独评估其稀疏质量；trailing loss 缺口依赖混合策略修复；部署无线上指标。
