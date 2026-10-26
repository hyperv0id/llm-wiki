---
title: "source-dao-2022-flashattention"
type: source-summary
tags:
  - source-summary
  - attention
  - efficiency
  - io-awareness
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# source-dao-2022-flashattention

FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness。Tri Dao、Daniel Y. Fu、Stefano Ermon（Stanford）、Atri Rudra（Buffalo）、Christopher Ré（Stanford），NeurIPS 2022；本库归档 arXiv:2205.14135v2（2022-06-24，34 页，含附录 A–E）[^src-dao-2022-flashattention]。

**问题与论点**：自注意力时间与显存随序列长度呈二次增长；近似注意力把 FLOP 降到近线性，却常无 wall-clock 加速。缺失的原则是 IO-aware：算法应计入 GPU HBM 与片上 SRAM 之间的读写次数。标准实现把 $N\times N$ 注意力矩阵写入 HBM，而注意力的大量步骤属于 memory-bound[^src-dao-2022-flashattention]。

**方法**：精确注意力 + tiling + 重算。前向把 Q、K、V 分块装入 SRAM，用 running max 与 rowsum 两个统计量在线重缩放并增量合并各块 softmax 贡献（Theorem 1 给出不变量与归纳证明），全程不落 $N^2$ 矩阵；反向只存输出、统计量与 PRNG 状态，注意力块在片上重算，dropout mask 由随机数状态重建。整个前向融合为一个 CUDA kernel。IO 复杂度从标准实现的 $\Theta(Nd + N^2)$ 降至 $\Theta(N^2d^2M^{-1})$（$M$ 为 SRAM 大小），并证明不存在对所有 $M$ 渐近更优的精确算法（Proposition 3）。扩展 block-sparse FlashAttention 将该项乘上稀疏比 $s$[^src-dao-2022-flashattention]。

**证据**（A100，作者报告）：BERT-large 训练 17.4 分钟对 MLPerf 1.1 记录 20.0 分钟（快 15%，Table 1）；GPT-2 对 HuggingFace 至多 3.5×、对 Megatron-LM 至多 2.0×（Table 2）；LRA 平均准确率 59.8、加速 2.4×，block-sparse 版 2.8× 快于全部被测近似方法（Table 3）；上下文扩到 4K 困惑度好 0.7 仍快 30%（Table 4）；Path-X 61.4%（16K）、Path-256 63.1%（64K）为 Transformer 首次超过随机（Table 6）；注意力算子显存随序列线性、比精确基线至多省 20×（图 3、Table 21）[^src-dao-2022-flashattention]。

**局限（作者自述）**：每个变体需手写 CUDA kernel、不可跨架构移植；分析限于单 GPU；块大小受 SRAM 约束，d=128 与小 SRAM 硬件上收益缩水[^src-dao-2022-flashattention]。

技术主体见 [[flash-attention]]。

[^src-dao-2022-flashattention]: [[source-dao-2022-flashattention]]
