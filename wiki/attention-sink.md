---
title: "Attention Sink (注意力汇聚)"
type: concept
tags:
  - llm
  - attention
  - kv-cache
  - streaming
created: 2026-08-30
last_updated: 2026-10-26
source_count: 3
confidence: medium
status: active
---

# Attention Sink（注意力汇聚）

Attention sink 指 LLM 中初始位置的 token 无论内容如何都获得不成比例的高注意力的现象。TriAttention 论文在 related work 中转述：丢弃这些 token 会使注意力分数失去稳定的"汇聚点"而性能退化；StreamingLLM 据此保留少量初始 sink token 加近期窗口，实现无限长流式输入[^src-triattention]。

与范数的关系：VATP 的观察（同转引自 TriAttention 论文）是 sink token 注意力虽高但 value 范数近零、对输出贡献小，单看注意力分数会高估其重要性[^src-triattention]。

在 TriAttention 的框架里，attention sink 型模式（远距离成峰）被解释为 [[attention-distance-preference|距离偏好曲线]] 的一种形态：部分注意力头的三角级数在远距离成峰，与近距离成峰（局部注意力）的头并存，峰位由 Q/K 中心决定[^src-triattention]。

MoBA 论文（Moonshot AI，2025）把 attention sink 型稀疏（初始 token 加近期窗口）归为 [[moba-attention]] 的特例：门控永远选初始块与最近块。该归类只说明固定门控能表达这种模式，不涉及 sink 现象为何产生[^src-moba-mixture-of-block-attention]。

Gated Attention 论文（Qwen Team，2025）给出一手量测：15A2B MoE 基线的首 token 注意力占比在层间平均 0.467，第 21 层达 0.83；在 SDPA 输出侧加 head-specific 逐元素 sigmoid 门控后降到 0.048（该层 0.04），层内最大激活同时由 1053 降到 94（Fig. 2、Table 4）[^src-gated-attention-for-large-language-models-2505-06708]。门控只加在 value 投影之后时，大激活同样被压到 125，首 token 占比却仍有 0.297——massive activation 不是 attention sink 的必要条件；把门控分数改成与输入无关（0.364）或去掉其稀疏性（NS-sigmoid，0.451），首 token 占比相对门控模型大幅回升、接近基线的 0.467（§4.3 称两种修改同时加重大激活与 sink）[^src-gated-attention-for-large-language-models-2505-06708]。该文把 sink 的消除归给逐 token、逐头、逐通道的输入相关稀疏，这一因果说法是作者假设，论文未给出解析论证。

> [!warning] 二手转述
> 关于 StreamingLLM 与 VATP 的两条论断转引自 [[source-triattention]] 的 related work 章节；原始出处 StreamingLLM（Xiao et al., ICLR 2024）与 VATP（Guo et al., EMNLP 2024）尚未 ingest，待其入库后应补一手引注并核对表述。

## 相关页面

- [[kv-cache-compression]] — sink token 与 StreamingLLM 所在的方法谱系
- [[attention-distance-preference]] — sink 型远距成峰的几何解释
- [[triattention]] — 提出该解释的论文
- [[moba-attention]] — 把 sink 型稀疏归为门控特例的块稀疏架构
- [[sdpa-output-gating]] — 用输入相关稀疏门控消除 sink 的逐层量测
- [[source-triattention]] — 源摘要

[^src-triattention]: [[source-triattention]]
[^src-gated-attention-for-large-language-models-2505-06708]: [[source-gated-attention-for-large-language-models-2505-06708]]
[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
