---
title: "Source: PagedAttention / vLLM — 分页 KV cache 内存管理 (SOSP 2023)"
type: source-summary
tags:
  - source
  - llm-serving
  - kv-cache
  - memory-management
  - efficient-inference
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Source: PagedAttention / vLLM（arXiv:2309.06180）

- **作者**：Woosuk Kwon\*, Zhuohan Li\*, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica（UC Berkeley / Stanford / Independent Researcher / UC San Diego）
- **时间**：2023-09-12 预印；SOSP '23（Koblenz），DOI 10.1145/3600006.3613165
- **链接**：https://arxiv.org/abs/2309.06180
- **本地**：`raw/pagedattention-vllm-2309-06180.pdf`（arXiv v1，16 页，含参考文献，无附录）
- **代码**：https://github.com/vllm-project/vllm

## 核心论点

LLM 服务的批大小被 KV cache 显存卡住。现有系统把一条请求的 KV cache 存成连续张量、按请求最大长度预分配，于是有三类浪费——给未来 token 预留的槽位、超出实际长度的内部碎片、buddy allocator 的外部碎片，实测有效显存只剩 20.4%–38.2%（Fig 2、§3.1）[^src-pagedattention]。PagedAttention 把 KV cache 切成固定大小的块，逻辑连续而物理可散落，注意力 kernel 按 block table 现场 gather；vLLM 在其上做按需分配、块级引用计数共享与抢占调度（§4）[^src-pagedattention]。

## 机制

每块容纳 $B$ 个 token（默认 16）在该层该头的 K/V，注意力按块改写为块内矩阵乘 + 跨块求和（式 4）；块表 entry 记物理块号与已填槽数，故只在生成时逐块增长，不再按上限预留（§4.1–§4.2）[^src-pagedattention]。物理块带引用计数：并行采样共享 prompt 块、束搜索沿候选树共享、共享前缀预置为常驻块，写共享块时触发块级 copy-on-write（§4.4）[^src-pagedattention]。抢占按 FCFS 与 all-or-nothing 淘汰，恢复走换出到 CPU 或重算（§4.5）[^src-pagedattention]。分布式用 Megatron-LM 式张量并行配单一集中式 block manager，各 worker 只存自己头的 KV（§4.6）[^src-pagedattention]。实现上三个融合 kernel 分别处理块写入、块读出 + 注意力、块复制（§5.1）[^src-pagedattention]。

## 证据

OPT-13B/66B/175B 与 LLaMA-13B，GCP A100，ShareGPT 与 Alpaca 长度合成负载（Table 1、§6.1）[^src-pagedattention]。ShareGPT 上维持同等延迟的请求率比 Orca (Oracle) 高 1.7×–2.7×、比 Orca (Max) 高 2.7×–8×、比 FasterTransformer 最高 22×；OPT-13B 同请求率并发批 30.42 条对 Orca (Oracle) 13.62 条（Fig 12、Fig 13）[^src-pagedattention]。块共享省内存 6.1%–9.8%（并行采样，Alpaca）与 37.6%–55.2%（束搜索），ShareGPT 上为 16.2%–30.5% 与 44.3%–66.3%；共享前缀 1.67×（80 token）/3.58×（341 token）（Fig 15–16）[^src-pagedattention]。代价是注意力 kernel 延迟高 20%–26%（Fig 18a）[^src-pagedattention]。

## 局限

只改放置、不改每 token 的 KV 字节数，也不淘汰任何 KV。收益依赖并发度：短序列加显存充裕（175B + Alpaca）时系统转为 compute-bound，领先幅度收窄（Fig 12f）[^src-pagedattention]。块大小是需扫描的超参，默认 16（§7.2）[^src-pagedattention]；§8 指出训练等形状静态的负载上引入间接寻址与非连续块反而更慢[^src-pagedattention]。

机制、数据流与内存浪费的算法见 [[paged-attention]]。

[^src-pagedattention]: [[source-kwon-2023-pagedattention]]
