---
title: "Attention Sink (注意力汇聚)"
type: concept
tags:
  - llm
  - attention
  - kv-cache
  - streaming
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: low
status: active
---

# Attention Sink（注意力汇聚）

Attention sink 指 LLM 中初始位置的 token 无论内容如何都获得不成比例的高注意力的现象。TriAttention 论文在 related work 中转述：丢弃这些 token 会使注意力分数失去稳定的"汇聚点"而性能退化；StreamingLLM 据此保留少量初始 sink token 加近期窗口，实现无限长流式输入[^src-triattention]。

与范数的关系：VATP 的观察（同转引自 TriAttention 论文）是 sink token 注意力虽高但 value 范数近零、对输出贡献小，单看注意力分数会高估其重要性[^src-triattention]。

在 TriAttention 的框架里，attention sink 型模式（远距离成峰）被解释为 [[attention-distance-preference|距离偏好曲线]] 的一种形态：部分注意力头的三角级数在远距离成峰，与近距离成峰（局部注意力）的头并存，峰位由 Q/K 中心决定[^src-triattention]。

> [!warning] 二手转述
> 本页论断转引自 [[source-triattention]] 的 related work 章节；原始出处 StreamingLLM（Xiao et al., ICLR 2024）与 VATP（Guo et al., EMNLP 2024）尚未 ingest，待其入库后应补一手引注并核对表述。

## 相关页面

- [[kv-cache-compression]] — sink token 与 StreamingLLM 所在的方法谱系
- [[attention-distance-preference]] — sink 型远距成峰的几何解释
- [[triattention]] — 提出该解释的论文
- [[source-triattention]] — 源摘要

[^src-triattention]: [[source-triattention]]
