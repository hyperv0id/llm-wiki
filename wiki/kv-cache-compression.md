---
title: "KV Cache Compression"
type: concept
tags:
  - llm
  - kv-cache
  - efficient-inference
  - attention
created: 2026-08-30
last_updated: 2026-10-26
source_count: 8
confidence: medium
status: active
---

# KV Cache 压缩（KV Cache Compression）

自回归 LLM 推理需要缓存全部历史 token 的 K/V 向量（KV cache）；长推理链生成数万 token 时，KV cache 随长度线性增长，成为内存与吞吐瓶颈。KV cache 压缩在固定内存预算下只保留部分 KV 对，核心难题是**重要性估计**：预测未来 query 会关注哪些 key[^src-triattention]。

## post-RoPE 方法分类（TriAttention 论文的口径）

论文将既有方法按信号来源分三类（本页对各家方法的描述转引自该论文的 related work，未回溯各原始论文）[^src-triattention]：

- **启发式**：StreamingLLM 保留少量初始 sink token（见 [[attention-sink]]）加近期滑动窗口，实现无限长流式输入；规则固定，无法适配内容相关的重要性。
- **注意力分数**：H2O 跨解码步累积注意力分数找 heavy-hitter token；SnapKV 在局部观察窗内聚合注意力分数预测生成期的重要 token；Scissorhands 基于"重要性持续性"假设用历史注意力指导驱逐；R-KV 用最近 query 的注意力打分并做冗余检测（面向推理模型）；LazyEviction 在观察窗内跟踪 token 重要性重现以延迟驱逐决策。
- **范数类**：VATP 指出注意力高但 value 范数近零的 token（如 sink token）对输出贡献小，用 value 范数修正注意力分数；KnormPress 同属该类。

## 共同限制：post-RoPE 观察窗口

论文对两类的批评都落在 post-RoPE 空间上[^src-triattention]：

- 注意力法：query 随位置旋转，方向未过时的只有最近的 query，可用的观察窗口极小；论文转述 Zhang et al. 2025 的结论——增大观察窗无效，性能在约 25 个 query 处见顶后回落。窗口内未被注意的 key 可能被永久驱逐，对 retrieval heads 中长期休眠、之后才必需的 token 尤其致命，在推理任务中破坏思维链。
- 范数法：只用向量幅值、丢弃方向信息；而注意力同时依赖范数与夹角，post-RoPE 方向与位置旋转纠缠、难以利用。

## pre-RoPE 路线

[[triattention]] 转向 pre-RoPE 空间：利用 [[qk-concentration]] 的稳定 Q/K 中心，经 [[attention-distance-preference|三角级数]] 直接从几何预测距离偏好，并叠加范数信号，无需观察近期注意力[^src-triattention]。

## 与其他长上下文路线的关系

KV 压缩与 [[context-window-extension]]（扩展位置编码以支持更长输入）解决的是长上下文的不同侧面：前者压推理期内存、面向长输出，后者扩输入窗口；两者可叠加。长输入下模型行为本身的退化见 [[long-context-scaling-gap]]。

正交的一条是训练期架构稀疏：[[moba-attention]] 用 top-k 门控选 KV 块，改 attention 计算结构而非推理期缓存淘汰；H2O、StreamingLLM、TOVA、FastGen、Quest 等属 decode 侧，与它互补[^src-moba-mixture-of-block-attention]。

副本数是另一条独立轴：[[multi-query-attention]] 不淘汰任何 KV 条目，只让全部 query head 共享同一份 $K,V$，解码期 K/V 的重读字节降为 $1/H$，KV cache 的元素数从 $HMd_k$ 压到 $Md_k$（$H=8$ 即 $8\times$）[^src-shazeer-2019-mqa]。

[[deepseek-sparse-attention|DSA]] 同属训练期架构稀疏，但不淘汰任何 KV：每个 query 在注意力时只读取 top-$k$ 个 MLA latent，主注意力复杂度由 $\mathcal{O}(L^2)$ 降到 $\mathcal{O}(Lk)$；代价是产生选择分数的 indexer 仍要对全部前序位置打分，保持 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。

[[native-sparse-attention|NSA]] 也属训练期稀疏且不淘汰 KV：每个 query 只读 $\lfloor (s-l)/d\rfloor+nl'+w$ 个 token（$l=32$、$d=16$、$l'=64$、$n=16$、$w=512$），64k 序列下为 5632 个等价 token，对应 11.6× 的解码预期加速（Table 4）；访存的节省依赖 GQA 组内共用同一批块，逐 head 独立选块时读量由组内并集决定（§2.1、§5.2）[^src-2502-11089-nsa]。

与压缩、稀疏并列的另一条轴是头共享：[[grouped-query-attention|GQA]] 不淘汰也不压缩任何 KV，把 $H$ 个 query head 分成 $G$ 组、每组共享一份 K/V，每步读取字节按 $G/H$ 缩放（$G=1$ 即 MQA 的 $1/H$，$G=H$ 即 MHA）；注意力算术量与 $G$ 无关、仍正比于 $H$，故算术强度提高 $H/G$ 倍（§2.2）。KV cache 容量同样缩到 MHA 的 $G/H$，且该比例不随模型规模漂移；MQA 固定砍到 $1/H$，头数随规模上调时砍得更狠。既有多头 checkpoint 可经 K/V 投影组内 mean pooling 加 5% 继续预训练转到 GQA：GQA-8 的 T5-XXL 平均 47.1 对 MHA-XXL 47.2，每样本 0.28 s 对 1.51 s（Table 1）[^src-ainslie-2023-gqa]。

[[trace-as-state|Trace as State]] 处理的是另一个约束：后出现的任务状态不能改变同一遍中已经形成的早期 token 表示。它把推理轨迹放在原文之前重新读取，不执行 KV 淘汰或压缩。论文 Appendix A 把位置、全部逐层 KV 条目及其他持久输入相关缓冲一并算入工作状态；“有限状态”并不等于“只保留一个固定大小向量”。[^src-trace-as-state]

作者在 Limitations 指出，轨迹前置可能降低多轮场景的 KV cache 复用；Appendix F 分别报告 cached input、missed input 与 output，不能据理论中的最坏情况内存差距宣称实际显存指数下降。完整开销口径见 [[trace-as-state-reproduction]]。[^src-trace-as-state]

## 放置轴（与压缩正交）

[[paged-attention]] 走的是另一条轴：不改每个 token 的 KV 字节数，也不淘汰任何 KV，只把 KV cache 切成固定大小的块按需分配、按引用计数共享，消掉预分配造成的碎片与重复——OPT-13B 上有效显存从 20.4%–38.2% 提到 96.3%（Fig 2）[^src-pagedattention]。它与本页的压缩淘汰、以及 MQA/GQA 式头共享（减少每 token 要存的 K/V 条数）可叠加：paging 决定这些 KV 落在哪、由谁共用，压缩与头共享决定每个 token 占多少字节。

## 相关页面

- [[triattention]] — pre-RoPE 压缩方法
- [[qk-concentration]] / [[attention-distance-preference]] — pre-RoPE 信号的两个前提
- [[attention-sink]] — sink token 现象
- [[rope]] — pre-RoPE/post-RoPE 区分的来源
- [[moba-attention]] — 训练期块稀疏架构（正交路线）
- [[multi-query-attention]] — 减少 K/V 副本数，不淘汰条目
- [[deepseek-sparse-attention]] — 训练期 token 粒稀疏，不淘汰 KV
- [[native-sparse-attention]] — 训练期块稀疏，三支路减少每步访存
- [[grouped-query-attention]] — 头共享：改 K/V 头数而非缓存内容
- [[paged-attention]] — 分页放置：块级分配与共享，不压字节、不淘汰
- [[source-triattention]] — 源摘要
- [[trace-as-state]] / [[conditional-state-update]] — 条件可用时机与缓存容量的区别

[^src-triattention]: [[source-triattention]]
[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-ainslie-2023-gqa]: [[source-ainslie-2023-gqa]]
[^src-trace-as-state]: [[source-trace-as-state]]
[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
[^src-pagedattention]: [[source-kwon-2023-pagedattention]]
