---
title: "Source: DeepSeek-V4.1-Flash — KV cache 压缩与 Causal Encoder-Decoder (arXiv:2609.19969)"
type: source-summary
tags:
  - source
  - attention
  - long-context
  - kv-cache
  - llm
  - mixture-of-experts
  - multimodal
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: DeepSeek-V4.1-Flash（arXiv:2609.19969）

作者为 DeepSeek-AI（集体署名），2026-09-17 发布（arXiv:2609.19969v1，cs.CL），为 51 页技术报告，未见 venue；论文链接为 https://arxiv.org/abs/2609.19969 ，checkpoints 见 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash ，本地见 `raw/deepseek-v4-1-flash-2609-19969.pdf`（51 页）。

## 核心论点

报告把长上下文服务成本拆成 prefill 计算、KV 存储（HBM/SSD）与搬运带宽三项，主攻存储与带宽：多模态 MoE，552B 骨干 + 196B Engram 参数，40 层（20 层编码器 + 20 层解码器），原生 1M 上下文，prefill 每 token 激活 8B、decode 16B（摘要、§2.1、Table 1）[^src-deepseek-v4-1-2609-19969]。全局 KV 降至 890 bytes/token，约为 V4-Flash 的 $1/4$；持久化 KV 约 $1/8$（摘要、§6）[^src-deepseek-v4-1-2609-19969]。

## 机制

CED 使解码器全局 KV 不由本层隐藏状态产生，而由编码器末层 $H^{L/2}$ 经逐层权重投影得到（式 1）；SWA 仍逐层计算，配合 decoder SWA bounded replay 仅重放最后 $n_{win}=128$ 个 token，prefill 复杂度为 $O(NL)\to O(NL/2+n_{win}L/2)$（§2.2）[^src-deepseek-v4-1-2609-19969]。CSA2 以三种静态模式（Full/Reindex/Reuse）跨层共享主 KV 与 indexer K，复用 Top-$k$ 下标；压缩器省去 CSA 的跨块重叠与绝对位置嵌入，indexer K 改由主 KV 投影；解码器首个 Full 层建立 $2{,}048\times8=16{,}384$ 位置候选池，供后续 Reindex 层打分（§2.3、§2.3.2）[^src-deepseek-v4-1-2609-19969]。主 KV 转为 FP4（E2M1 + 每 16 通道一个 E4M3 scale，在 RoPE 之后量化），SWA KV 保留 FP8（§2.4.4）[^src-deepseek-v4-1-2609-19969]。其余包含 Single-Pass mHC、Engram、DSpark 与 head-wise Muon（§2.4、§2.5）[^src-deepseek-v4-1-2609-19969]。

训练侧不再需要 dense 预热：从 64K 序列起直接训稀疏注意力，45T token（§1、§4.2.2）[^src-deepseek-v4-1-2609-19969]。

## 证据

参考系为 V4-Flash：全局 KV 为 $1/4$、相对 V1 为 $1/437$（Figure 1b）；持久化 KV 为 $1/8$，来自「SWA KV 不再落盘」（约占原容量一半）与「全局 KV $1/4$」（§3.2.1）[^src-deepseek-v4-1-2609-19969]。4K→1M 的 decode FLOPs 只增 $1/4$（Figure 2）[^src-deepseek-v4-1-2609-19969]。Base 对照（Table 1，V4.1 / V4-Pro / V4-Flash）：MMLU-Pro 74.1 / 73.5 / 68.3、HumanEval 79.4 / 76.8 / 69.5、LongBench-V2 45.2 / 51.5 / 44.7、SimpleQA-Verified 42.3 / 55.2 / 30.1[^src-deepseek-v4-1-2609-19969]。后训练（Table 3，Max 档）：Codeforces 3471（Pro 3348、Flash 3289）、DeepSWE v1.1 74.2（62.7、54.4）、Terminal-Bench 2.1 90.6（87.9、82.7）、HLE 36.8（39.1 为 text-only 子集）[^src-deepseek-v4-1-2609-19969]。

## 局限

后训练无算法创新，收益全部归因于数据与环境管线（§5.1）[^src-deepseek-v4-1-2609-19969]。CED 与 CSA2 均未与 V4 同台消融，退化只写「negligible」而无数字；两类未刻画的鲁棒性边界为 CSA2 的选择错误与 SWA 状态的近似重建（§6）[^src-deepseek-v4-1-2609-19969]。Base 表中长上下文与事实性两项低于 V4-Pro（LongBench-V2 45.2 对 51.5、SimpleQA 42.3 对 55.2），正文「comparable」判断未覆盖这两项。细节见 [[csa2]]、[[ced]]。

[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]
