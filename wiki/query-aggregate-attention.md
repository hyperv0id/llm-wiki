---
title: "Query-Aggregate Attention"
type: technique
tags:
  - traffic-forecasting
  - attention
  - spatio-temporal
  - rope
  - transformer
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# Query-Aggregate Attention

**Query-Aggregate Attention** 是 [[stunet|STUNet]] 的时空融合骨干：在 spatial tokens 与 temporal tokens **语义空间不对齐**时，把融合拆成「查询上下游位置 → 再聚合因果相关传感器信息」两步，并用两套 RoPE 分别服务查询与聚合，而不是对两类 token 做对称 full attention[^src-stunet]。

## 为何不能 full attention

空间 token 来自邻接 patch（位置 = 矩阵行列），时间 token 来自传感器×时间 patch（位置 = 传感器 id + 时间步）。直接混算会破坏各自相对位置语义；消融中 “w/ full attention” 在 SD/GBA/GLA 均明显变差[^src-stunet]。

## 两步

**Query attention.** Temporal 为 $Q$，spatial 为 $K/V$。RoPE 设计：时间 token 两半都按传感器索引旋转，用于在关系图中查相对行/列（源/汇）；空间 token 两半分别按 patch 的行、列索引旋转，保留 2D 结构位置。内积后每个时间 token 聚到其上下游结构信息[^src-stunet]。

**Aggregate attention.** 在 query 输出上做 temporal 自注意力：一半 RoPE 管传感器维（聚相关传感器时序），一半管时间 patch 维（时滞）。多层堆叠；**spatial tokens 跨层复用**（每层 $E_s$ 相同），体现“结构基固定、时间在其上查询”的归纳偏置[^src-stunet]。

直觉：显式模拟交通沿拓扑的上游→下游追溯与信息汇聚，而不是让模型从混叠 embedding 里暗中学边[^src-stunet]。

## Related

- [[stunet]] · [[source-stunet]] · [[spatial-tokenizer-adjacency-patches]]
- [[patchstg]]（depth/breadth dual attention：地理 patch 内/间，目标是效率而非跨网结构基）

[^src-stunet]: [[source-stunet]]
