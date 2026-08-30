---
title: "KV Cache Compression"
type: concept
tags:
  - llm
  - kv-cache
  - efficient-inference
  - attention
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
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

## 相关页面

- [[triattention]] — pre-RoPE 压缩方法
- [[qk-concentration]] / [[attention-distance-preference]] — pre-RoPE 信号的两个前提
- [[attention-sink]] — sink token 现象
- [[rope]] — pre-RoPE/post-RoPE 区分的来源
- [[source-triattention]] — 源摘要

[^src-triattention]: [[source-triattention]]
