---
title: "STUNet"
type: entity
tags:
  - traffic-forecasting
  - spatio-temporal
  - generalization
  - zero-shot
  - transformer
  - explicit-graph-modeling
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# STUNet

**STUNet**（Spatio-Temporal Unified Network）是面向[[traffic-forecasting|交通预测]]的显式时空统一框架，由 Chen、Tu 等（浙大 / SUPCON）提出于 KDD 2026[^src-stunet]。标题命题即核心卖点：**Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting**——用与时间解耦的统一空间 token 支撑跨路网泛化[^src-stunet]。

## 诊断

隐式空间建模（STGNN 消息传递、attention 构图、node-wise embedding）在训练中把空间结构与时序观测缠在一起；时间波动污染空间表示，换传感器集合/邻接/城市即崩。作者主张空间应由**道路拓扑**决定，应显式、可冻结、可跨网复用[^src-stunet]。

## 架构

| 组件 | 作用 |
|------|------|
| [[spatial-tokenizer-adjacency-patches\|Spatial tokenizer]] | 邻接矩阵切 patch → 统一 spatial tokens；Stage 1 AE + 节点置换增强，Stage 2 **冻结** |
| Temporal tokenizer | 时间 patch + ToD/DoW embedding（PatchTST 风格） |
| [[query-aggregate-attention\|Query-Aggregate Attention]] × L | Query：用空间 token 定位上下游；Aggregate：在 temporal 上聚合同源与相关传感器 |
| MLP head | 仅用融合 temporal 输出做多步预测 |

Code：`github.com/JimmyChen6/STUNet`[^src-stunet]。

## 泛化证据（论文主实验 RQ1）

**Train on network A → zero-shot test on B**，流数据 SD/GBA/GLA 互不重叠，速度数据 METR-LA/PEMS-BAY/SZ-TAXI；graph-free 基线去掉 node embedding。相对 DLinear、PatchTST、STID、iTransformer、STGCN、STWave、[[patchstg|PatchSTG]]，STUNet **全部迁移设定与指标最优**[^src-stunet]。示例 SD→GBA 平均 horizon：MAE 34.46 vs PatchSTG 37.98[^src-stunet]。

In-domain 与强基线持平并常拿下最佳 RMSE；消融显示 spatial tokens、tokenizer 预训练、**冻结**、query-aggregate（非 full attention）、邻接置换增强均为必要[^src-stunet]。Spatial token 可分直路/Y 形/环（ARI 0.96）[^src-stunet]。

## 与邻近工作

| 工作 | 关系 |
|------|------|
| [[patchstg\|PatchSTG]] | 同为 patch + Transformer 做大规模交通；PatchSTG 对**传感器地理点**做不规则空间 patch 提效，STUNet 对**邻接矩阵**做 patch 以显式结构 + 跨网零样本 |
| [[stop\|STOP]] | 同为结构 OOD/跨图；STOP 用集中式 messaging **阻断** node-to-node；STUNet **显式 token 化**邻接并 query 上下游 |
| [[opencity\|OpenCity]] / [[unist\|UniST]] 等 | 多城预训练 foundation；STUNet 是单网训练 + 显式结构迁移，非大规模预训练 FM |
| STID 等 node embedding | 换网需重训 embedding；STUNet 空间 token 来自图结构，可直接接到新邻接 |

## Related

- [[source-stunet]] — 源摘要
- [[spatial-tokenizer-adjacency-patches]] · [[query-aggregate-attention]]
- [[traffic-forecasting]] · [[ood-generalization]] · [[spatio-temporal-ood-learning]]
- [[patchstg]] · [[stop]] · [[spatio-temporal-foundation-model]]

[^src-stunet]: [[source-stunet]]
