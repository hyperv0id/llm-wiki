---
title: "Mixtral of Experts (arXiv:2401.04088)"
type: source-summary
tags:
  - mixture-of-experts
  - sparse-activation
  - large-language-models
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# source: Mixtral of Experts (Jiang et al., Mistral AI, 2024)

**归档**：`raw/mixtral-2401-04088.pdf`（arXiv:2401.04088v1，2024-01-08，Apache 2.0 开源权重）

Mistral AI 提出 Mixtral 8x7B：基于 Mistral 7B 将每层 FFN 换为 8 专家 MoE 层，router 取线性层 Top-2 logits 做 softmax，每 token 加性合并 2 个 SwiGLU 专家输出；总参 47B、激活 13B，32k 上下文预训练，多语言数据上采样[^src-mistral-2024-mixtral]。

核心结果：同一评测管线重跑下，以少 5 倍激活参数在多数基准持平或超过 Llama 2 70B（MMLU 70.6% 对 69.9%，MATH 28.4% 对 13.8%，HumanEval 40.2% 对 29.3%，Table 2），与 GPT-3.5 互有胜负（Table 3）；法德西意四语言三项基准全超 Llama 2 70B（Table 4）；passkey 检索达 100%，proof-pile 困惑度随上下文单调下降（Figure 4）；BBQ 准确率 56.0% 对 51.5%（Figure 5）。Instruct 版经 SFT+DPO 达 MT-Bench 8.30，为截至 2023-12 最佳开源权重模型（§4）[^src-mistral-2024-mixtral]。

路由分析（§5）显示，专家分工按句法（'self'、缩进 token）而非主题（The Pile 八个子集分布近均匀，Figure 7/8）；连续 token 同专家率在第 15 层达 61.6%–67.0%（Table 5），论文由此指出 Expert Parallelism 过订阅与缓存两个工程含义。论文未提及辅助负载均衡损失，路由均衡是 router 学出的观测结果（接近 1/8 均匀线）[^src-mistral-2024-mixtral]。

效率分析只计激活参数，服务内存仍按 47B 总参数；预训练数据量与硬件未披露。评测口径与 Llama 2 论文有两处差异（MBPP 人工核验子集、TriviaQA 无 Wikipedia 上下文），闭源对照仅给出 GPT-3.5/Claude-2.1/Gemini Pro 的时间点数字[^src-mistral-2024-mixtral]。

机制与数据流见 [[mixtral-of-experts]]。

[^src-mistral-2024-mixtral]: [[source-mistral-2024-mixtral]]
