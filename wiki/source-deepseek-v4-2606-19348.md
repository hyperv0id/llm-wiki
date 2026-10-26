---
title: "Source: DeepSeek-V4 — 混合压缩注意力与百万 token 上下文 (arXiv:2606.19348)"
type: source-summary
tags:
  - source
  - attention
  - long-context
  - llm
  - mixture-of-experts
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: DeepSeek-V4（arXiv:2606.19348）

- **作者**：DeepSeek-AI（集体署名）
- **时间**：2026-04-26（arXiv:2606.19348v1，cs.CL）；preview 版技术报告，未见 venue
- **链接**：https://arxiv.org/abs/2606.19348 ；checkpoints：https://huggingface.co/collections/deepseek-ai/deepseek-v4
- **本地**：`raw/deepseek-v4-2606-19348.pdf`（58 页）

## 核心论点

DeepSeek-V4 系列两个 MoE 模型：Pro 1.6T 总参/49B 激活、Flash 284B/13B，原生支持 1M token 上下文；三项升级为混合注意力（CSA+HCA）、mHC 残差改造、Muon 优化器（摘要、§2）[^src-deepseek-v4-2606-19348]。

## 机制

CSA 把每 4 个 token 的 KV 压成 1 个 entry（重叠双支路，净压缩率 $1/4$）再跑 DSA top-$k$（512/1024）；HCA 每 128 个 token 压 1 个 entry、对全部条目 dense 注意力；两者交错并共享滑窗、QK RMSNorm、partial RoPE 与 attention sink（§2.3）[^src-deepseek-v4-2606-19348]。预训练 Flash 32T / Pro 33T token，序列 4K→1M 渐进，dense 预热后 64K 起切稀疏（§4.2.2）；后训练为领域专家 GRPO + 全词表 on-policy distillation（10+ 教师），并做 FP4 QAT（§5.1、§5.2）[^src-deepseek-v4-2606-19348]。

## 证据

1M 上下文下，Pro 单 token FLOPs 为 V3.2 的 27%、KV cache 为 10%；Flash 为 10%/7%；KV cache 约为 BF16 GQA8 基线的 2%（§1、§2.3.4、Figure 1 右）。Base 对照（Table 1）：LongBench-V2 40.2→44.7→51.5、Simple-QA verified 28.3→30.1→55.2。Pro-Max（Table 6）：SimpleQA-Verified 57.9（开源最高，落后 Gemini-3.1-Pro 的 75.6）、LiveCodeBench 93.5、Codeforces 3206、MRCR 1M 83.5；推理落后前沿闭源约 3–6 个月（§1）[^src-deepseek-v4-2606-19348]。

## 局限

架构因求稳偏复杂，计划后续精简；Anticipatory Routing 与 SwiGLU clamping 有效机理未明（§6）[^src-deepseek-v4-2606-19348]。报告未提供注意力组件消融（CSA/HCA 配比、$m$、$k$ 无敏感性），效率数字为 Figure 1 估计值，长上下文对照基于不同系统。机制细节见 [[compressed-sparse-attention]]、[[heavily-compressed-attention]]、[[deepseek-v4-hybrid-attention]]。

[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
