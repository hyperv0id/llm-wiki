---
title: "MiniMax Sparse Attention"
type: source-summary
tags:
  - attention
  - sparse-attention
  - long-context
  - llm
  - gpu-kernel
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# MiniMax Sparse Attention（MSA）

MiniMax（Xunhao Lai 等，MiniMax + 北京大学/NVIDIA/浙江大学等），arXiv:2606.13392v2（2026-06-12）。针对超长上下文（数十万到百万 token 的 agentic 负载）下 softmax 注意力的平方成本，论文在 GQA 上增加块稀疏两支路，遵循 Occam's razor，消融后只保留必要组件，可从零稀疏预训练，也可从 GQA checkpoint 转换（§1）[^src-minimax-sparse-attention-2606-13392]。

机制（§3）：Index Branch 为每个 GQA 组配置一个索引 query 头，所有 GQA 组共享一个索引 key 头；token 打分后经块内 max 池化得到块分，逐组独立选取 top-$k$ 块（$B_k=128$、$k=16$），强制纳入包含 query 的 local block；Main Branch 仅在选中块的因果可见 token 上计算精确 softmax，每 query 预算 $kB_k=2048$ token，与 $N$ 无关。新增参数仅为索引 $Q/K$ 两个投影矩阵。训练采用 KL 对齐损失（教师为组内 $G$ 个主支路分布在概率级平均并设 stopgrad）、索引输入梯度 detach、indexer warmup 与强制 local block（§3.2）。kernel 侧采用 exp-free top-k、KV-outer 遍历（算术强度约 $\tfrac{2}{3}B_k$ 对 Q-outer 的约 $G$）、槽位预分配与两阶段合并（§4）。

证据（§5）：109B 总参/6B 激活 MoE、3T token 预算。Table 2 上 MSA-PT 与 MSA-CPT 对 GQA 基线整体可比，PT 在视频与长上下文检索多数占优；Figure 4 在 1M 上下文报每 token 注意力 FLOPs 降 28.4×，H800 实测 prefill 14.2×、decode 7.6×；Table 3 长上下文扩展后 HELMET-128K Δ−0.60、RULER-128K Δ+0.12。

局限在于残差长程检索缺口待解决（§7）。因索引/gather/负载均衡开销，运行时加速小于 FLOPs 减少（§5.4）。实验未与其他稀疏注意力方法同台，§6 只做定性定位，消融多在 10B 试点模型或子集评测上（附录 B/C）。

## 相关页面

[[minimax-sparse-attention]] 涵盖机制、数据流、kernel 与全部实验数值；[[native-sparse-attention]] · [[moba-attention]] · [[deepseek-sparse-attention]] · [[compressed-sparse-attention]] · [[longformer-attention]] 为稀疏注意力谱系对照。

[^src-minimax-sparse-attention-2606-13392]: [[source-minimax-2026-sparse-attention]]
