---
title: "Q/K Concentration (pre-RoPE Q/K 集中)"
type: concept
tags:
  - llm
  - attention
  - rope
  - representation-geometry
  - kv-cache
created: 2026-08-30
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# Q/K Concentration（Q/K 集中）

Q/K concentration 是 TriAttention 论文报告的经验现象：在施加 RoPE 旋转之前的 pre-RoPE 空间中，跨大多数注意力头，Q 与 K 向量高度集中于一个固定的非零中心，且该结构跨 token 位置、跨输入内容保持稳定[^src-triattention]。论文认为，由于 pre-RoPE 向量尚未经过位置旋转，这种稳定性是内在的而非巧合[^src-triattention]。

## 量化：Mean Resultant Length

集中度用方向统计学的 Mean Resultant Length 度量：$R=\|\mathbb{E}[q]\|/\mathbb{E}[\|q\|]$，对每个频率带 $f$ 单独计算 $R_f$。$R\to 1$ 表示所有向量指向同一方向（完全集中），$R\to 0$ 表示方向均匀分散[^src-triattention]。

论文报告的测量（均在其设置内）[^src-triattention]：

- Qwen3-8B 的 1152 个头（36 层 × 32 头）中，绝大多数 $R$ 接近 1.0；
- Math（MATH-500）、Coding、Chat 三个域上平均 MRL 几乎相同（0.977–0.980），约 90% 的头 $R>0.95$，论文据此称该现象是 model-intrinsic 而非领域特有；
- MLA 架构（GLM-4.7-Flash，940 头）上 $R>0.95$ 的头占 96.6%，高于 GQA 架构（Qwen3-8B）的 84.7%；
- 重建实验（见 [[attention-distance-preference]]）在 Qwen3、Qwen2.5、Llama3 三个模型族上方向一致。

## 发现方式的注记

作者在 ICML 评审回复中说明：该现象最初在数学数据上观察到，随后在 code 与对话数据上验证同样成立[^src-triattention-openreview]。评审人曾追问现象是否只在 AIME 类数据上成立，上述跨域测量是对该问题的回应[^src-triattention-openreview]。

## 与注意力行为的关系

集中使注意力 logit 可近似为只依赖 Q-K 距离的三角级数，中心决定系数与距离偏好曲线（[[attention-distance-preference]]）；这正是 [[triattention]] 用 Q 中心估计 key 重要性、替代 post-RoPE 注意力观察的基础，也是 [[kv-cache-compression]] 中 pre-RoPE 路线的出发点[^src-triattention]。

注意边界：约 10–15% 的头集中度较低（GQA 架构下 $R\le 0.95$ 的头约 15%），论文对这些头用 $(1-R_f)$ 加权的范数分数补位，而非假设三角级数处处成立[^src-triattention]。

## 相关页面

- [[attention-distance-preference]] — 集中导致的可预测距离偏好
- [[triattention]] — 利用该现象的 KV cache 压缩方法
- [[rope]] — 旋转位置编码；pre-RoPE/post-RoPE 的区分由其定义
- [[kv-cache-compression]] — 问题域
- [[source-triattention]] — 源摘要

[^src-triattention]: [[source-triattention]]
[^src-triattention-openreview]: [[source-triattention-openreview]]
