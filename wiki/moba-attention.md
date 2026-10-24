---
title: "MoBA: Mixture of Block Attention"
type: technique
tags:
  - attention
  - sparse-attention
  - mixture-of-experts
  - long-context
  - llm
created: 2026-10-08
last_updated: 2026-10-24
source_count: 3
confidence: medium
status: active
---

# MoBA: Mixture of Block Attention

Moonshot AI（Kimi 团队）2025-02 提出的长上下文注意力架构（arXiv:2502.13189）：把 [[mixture-of-experts|MoE]] 的 top-k 门控原则用于 attention，已部署于 Kimi 的长上下文请求[^src-moba-mixture-of-block-attention]。

## 问题

标准 attention 的计算量随序列长度二次增长，是扩展 LLM 有效上下文的主要开销。MoBA 的动机是两条既有路线各有代价[^src-moba-mixture-of-block-attention]：

- **强偏置结构**（sink attention、sliding window attention）：稀疏模式是预定义的、任务特定，不保证适配内容相关的重要性。
- **线性 attention**（Mamba、RWKV、RetNet 一类）：softmax 换成线性近似，与既有 Transformer 差异大，转换成本高或需从头训练；复杂推理任务上的有效性证据有限。

MoBA 的设计原则是 "less structure"：不预定义稀疏模式，让模型自己学该注意哪里。

## 机制

### 块即专家，query 自任门控

上下文按块大小 $B$ 切成 $n$ 个块 $I_i$。对 query $\bm{q}$，块 $i$ 的亲和分数用 query 与块内 key 均值的内积计算：

$$s_i = \langle \bm{q},\ \mathrm{mean\_pool}(\bm{K}[I_i])\rangle$$

门控 $g_i \in \{0,1\}$ 取 top-k，query 只对 $g_i=1$ 的块做 attention。与标准 MoE 的两点差异：门控输入是 query 本身而非额外网络，不引入新参数；专家是 KV 块而非 FFN 子网络，切分沿上下文长度维度而非 FFN 中间维度[^src-moba-mixture-of-block-attention]。

### 因果性的两个设计

1. **不选未来块**：位置在 query 之后的块 $s_i = -\infty$、$g_i = 0$。
2. **当前块强制选择 + causal mask**：query 所在块的 mean pooling 会混入未来 token 的信息，因此当前块强制 $g_i=1$ 且块内 attention 加 causal mask，作用相当于 MoE 的 shared expert（静态规则叠加在动态路由上）[^src-moba-mixture-of-block-attention]。

### 与静态稀疏模式的关系

sliding window attention 与 attention sink（现象见 [[attention-sink]]）都是 MoBA 的特例：前者等价于门控永远选最近的块，后者等价于永远选初始块加最近块；反过来，MoBA 的门控随输入变化，不限于固定模式[^src-moba-mixture-of-block-attention]。

### 与 full attention 的无缝切换

MoBA 是 full attention 的 drop-in 替代：参数量不变，top-k 覆盖全部块时退化为 full attention；每层可在初始化或训练中动态选择两种模式[^src-moba-mixture-of-block-attention]。

### 实现

五步流程，无需自定义 CUDA kernel：门控加因果掩码确定 query 到块的分配；按所分配的块重排 query；块级 attention 用变长 FlashAttention 计算（当前块 causal、历史块 non-causal，分开计算）；输出重排回原序；用 online softmax（tiling）合并跨块输出[^src-moba-mixture-of-block-attention]。

## 证据

以下均出自论文实验，各条目写明设置。

### 缩放定律（8K 训练）

5 个模型（568M–2.1B，Chinchilla 足量 token），序列长 8K，块 512，top-3，稀疏度至多 $1-\frac{512\times 3}{8192}=81.25\%$。验证集 LM loss 与 full attention 的差保持在 1e-3 内；但 32K 外推的 trailing loss，正文表格（seqlen=32K、last 2K）拟合为 MoBA $1.546\times C^{-0.108}$、full $1.464\times C^{-0.097}$，纯 MoBA 在尾部 token 上落后（图 3 标题写 last 1K、表格写 last 2K，论文内部口径不一）[^src-moba-mixture-of-block-attention]。

附录 A.1 按位置分段评测（消除自然数据偏短上下文的偏差）：trailing token 贡献了稀疏与 full attention 差距的大部分，长上下文缩放研究可聚焦 trailing token[^src-moba-mixture-of-block-attention]。

### 块粒度消融

1.5B 模型、32K 上下文，块数 8 到 128（top-k 相应 2 到 32），稀疏度固定 75%。更细的块在同一稀疏度下损失更低，最粗设置（8 选 2）与更细设置间相差 1e-2[^src-moba-mixture-of-block-attention]。

### 混合训练与逐层混合

1.5B、30B token、32K 上下文（块 2048、top-3）[^src-moba-mixture-of-block-attention]：

- **MoBA/full 两阶段**：前 90% token 用 MoBA、后 10% 切 full attention，position-wise loss 几乎回到 full 水平；切换时未观察到明显 loss spike。
- **逐层混合（面向 SFT）**：纯 MoBA 的 SFT loss 偏高，原因未直接验证：SFT 的 loss masking（prompt 不计损失）只从回答 token 回传梯度，对稀疏 attention 构成稀疏梯度问题；把最后几层换成 full attention 可显著降低 SFT loss。

### 1M 上下文评测

从 Llama 3.1 8B 出发持续预训练到 1M 上下文（256K 阶段起点用位置插值），MoBA 训 100B token（块 4096、top-12，1M 时稀疏度至多 95.31%），最后 3 层保留 full attention、其余 29 层换 MoBA。评测中 MoBA 只用于 prefill，生成期切回 full attention。RULER 128K（稀疏度 62.5%）0.7818 vs full 0.7849；GSM8K、BBH 等短任务分数相当[^src-moba-mixture-of-block-attention]。

### 效率

只比较 attention 层前向：1M token prefill 加速至多 6.5 倍；固定块数与 top-k、按比例增大块大小以维持 95.31% 稀疏度（配合 query head 级张量并行），10M token 时相对 FlashAttention 加速 16 倍，计算复杂度次二次，加速来自块稀疏机制与 MoE/FlashAttention 联合实现[^src-moba-mixture-of-block-attention]。

## 范围与边界

- Kimi 部署仅论文一条来源，无细节与线上指标[^src-moba-mixture-of-block-attention]。
- 下游评测中 MoBA 仅用于 prefill，decode 用 full attention——稀疏 attention 在生成期的质量未在论文中单独评估[^src-moba-mixture-of-block-attention]。
- 纯 MoBA 训练存在 trailing loss 缺口，需混合策略修复；混合配比（90/10 两阶段、末 3 层 full）是其实验设置下的选择[^src-moba-mixture-of-block-attention]。
- MiTA 论文把块路由的专家构造判为「仅位置驱动」的刚性划分，主张专家本身也应随内容形变；该判断是分析论证，未与 MoBA 同台评测[^src-2602-01219-mita]。
- 选择粒度与 [[deepseek-sparse-attention|DSA]] 不同：MoBA 以块为单位（$N/n$ 个专家），DSA 以单个 MLA KV latent 为单位（$N$ 个宽 $k$ 的专家），代价是 DSA 的打分步骤仍为 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。
- Quest 可视为更小块加 min/max pooling 块表示的 MoBA，LongHeads 可视为 top-1 MoBA（出自该 related work）；H2O、StreamingLLM、TOVA、FastGen 属 decode 侧 KV 压缩，与 MoBA 的训练期架构稀疏正交（见 [[kv-cache-compression]]）[^src-moba-mixture-of-block-attention]。

## 相关页面

- [[mixture-of-experts]] — MoBA 直接借用的 top-k 门控原则
- [[attention-sink]] — 被 MoBA 论文归为静态稀疏特例之一的现象
- [[kv-cache-compression]] — 推理期 KV 压缩路线，与 MoBA 训练期稀疏正交
- [[source-moba-mixture-of-block-attention]] — 源摘要
- [[mita-attention]] — 形变 expert 替代定长块，$N$-to-$m$ 路由（论文自述动机）
- [[deformable-fast-weight-experts]] — 刚性块与形变专家的对照
- [[fast-weight-scaling-taxonomy]] — MoBA 在五维分类中的格位
- [[deepseek-sparse-attention]] — token 粒度的 top-$k$ 对照

[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
