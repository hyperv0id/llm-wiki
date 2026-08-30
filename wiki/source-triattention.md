---
title: "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"
type: source-summary
tags:
  - llm
  - kv-cache
  - attention
  - rope
  - efficient-inference
  - icml-2026
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# TriAttention（arXiv 版）

TriAttention 是 MIT、NVIDIA 与浙江大学团队合作提出的 KV cache 压缩方法（前三位作者共同一作：Weian Mao、Xi Lin、Wei Huang），ICML 2026 接收（评审记录见 [[source-triattention-openreview]]）[^src-triattention]。

问题：长推理生成数万 token，KV cache 线性增长成为内存瓶颈；主流压缩方法用近期 post-RoPE query 的注意力分数估计 KV 重要性，但 query 随位置旋转，方向可用的代表性 query 很少，top-key 选择不稳定[^src-triattention]。

核心观察（Q/K concentration）：在 pre-RoPE 空间中，多数注意力头的 Q/K 向量集中于固定非零中心，跨位置、跨输入稳定；按 Mean Resultant Length 度量，Qwen3-8B 约 90% 的头 R>0.95，Math/Coding/Chat 三域 MRL 0.977–0.980，MLA 架构（GLM-4.7-Flash，940 头）R>0.95 的头占 96.6%[^src-triattention]。

机制：Q/K 集中时代入 RoPE 公式，注意力 logit 近似为只依赖 Q-K 距离的三角级数，系数由 Q/K 中心决定，距离偏好因此可由中心预测（三个模型族的逐头重建 Pearson r 均值 >0.5）。方法据此用 Q 中心构造三角级数分数 $S_{\text{trig}}$ 按位置给 key 打分，叠加按 $(1-R_f)$ 加权的范数分数 $S_{\text{norm}}$，在几何间隔的未来偏移集 $D=\{1,2,4,\dots,2^{16}\}$ 上平均，每 128 token 触发一次 top-B 剪枝，GQA 下先逐头 z-score 归一再取最大聚合[^src-triattention]。

证据：AIME25（32K 生成、Qwen3-8B）与 Full Attention 同精度（40.8%）时吞吐 2.5 倍或 KV 内存省 10.7 倍；同预算下 AIME25 精度 32.9%，约为 R-KV（17.5%）与 SnapKV（20.0%）的 1.6–1.9 倍；自建递归状态查询基准（DFS 模拟）测内存保持，depth≤16 与 Full Attention 相当，R-KV 在 depth 16 从约 61% 骤降至 31%；附录另报 LongBench 16 子任务平均 48.1（16 项中 11 项最优）与 RULER 66.1[^src-triattention]。

局限（论文自述）：三角级数计算与剪枝尚无专门 kernel；评估拟扩展至 coding 与 agentic 任务；head-specific 预算留作未来工作[^src-triattention]。

[^src-triattention]: [[source-triattention]]
