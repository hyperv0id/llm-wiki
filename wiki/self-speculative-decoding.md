---
title: "Self-speculative Decoding"
type: technique
tags:
  - llm
  - inference
  - speculative-decoding
  - decoding
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# Self-speculative Decoding

**Self-speculative decoding（自推测解码）** 用**同一模型**的额外预测头起草未来 token，再由 next-token 头（或验证路径）并行核验，从而在保持（近似）原分布的前提下加速自回归生成。无需单独 draft 小模型[^src-gloeckle-2024-multi-token-prediction]。

## 与标准 Speculative Decoding 的关系

| | Speculative decoding (Leviathan et al.) | Self-speculative |
|--|----------------------------------------|------------------|
| Draft 来源 | 外部小模型 | 同模型多 head / 线性头 |
| 额外模型 | 需要 | 不需要 |
| 典型实现 | 小模型起草 + 大模型验证 | Blockwise parallel decoding (Stern et al., 2018)；Medusa tree attention (Cai et al., 2024) |

Gloeckle et al. 的 [[multi-token-prediction|multi-token prediction]] 预训练让额外 head 在**预训练阶段**就学会准确的多步草稿，比仅在 next-token 模型上 finetune 多头更利于吃满推测加速[^src-gloeckle-2024-multi-token-prediction]。

## 实证（Gloeckle et al., 7B）

- **4-token** 模型，greedy self-speculative：code 约 **3.0×** 吞吐（平均约 2.5/3 接受），text 约 **2.7×**；相对加速在 batch size 1–40 大致稳定。
- **8-byte** 预测模型：code 上约 **6.4×**（使用 8 heads），可抵消字节序列更长的推理劣势[^src-gloeckle-2024-multi-token-prediction]。

理论最大加速受 head 数与接受率限制；接受率取决于 head 质量，而 multi-token **预训练**正是提高接受率的关键。

## 相关

- [[multi-token-prediction]] — 产出可用多头的预训练目标
- [[source-gloeckle-2024-multi-token-prediction]] — 源论文与测速表

## 引用

[^src-gloeckle-2024-multi-token-prediction]: [[source-gloeckle-2024-multi-token-prediction]]
