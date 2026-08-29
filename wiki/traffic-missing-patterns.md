---
title: "交通缺失模式四分类（SRTR/SRTC/SCTR/SCTC）"
type: concept
tags:
  - spatiotemporal-imputation
  - missing-patterns
  - traffic
  - benchmark
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# 交通缺失模式四分类（SRTR/SRTC/SCTR/SCTC）

Guo、Wei 等人（arXiv:2412.04733v2）将交通数据缺失按"缺失发生在哪个维度（空间 S / 时间 T）"与"缺失位置随机（R）还是连续（C）"两个正交维度分为四类[^src-guo-imputation-evaluation]：

| 模式 | 结构 | 论文给出的现实成因（题注口径） |
|------|------|------------------------------|
| SRTR（Spatial Random Temporal Random） | 时空均随机 | 随机信号或网络中断 |
| SRTC（Spatial Random Temporal Continuous） | 空间随机、时间连续 | 部分传感器在一段时间内设备故障或断网 |
| SCTR（Spatial Continuous Temporal Random） | 空间连续、时间随机 | 一组设备因区域内网络中断等因素故障 |
| SCTC（Spatial Continuous Temporal Continuous） | 时空均连续 | 一组设备在一段时期内区域断电等 |

该分类是同一评测的实践导向（practice-oriented）产物：四类对应不同的掩码构造协议——SRTR 以概率 α 对每个元素独立置零；SRTC 先把时间轴切成不重叠 patch 再按 patch 置缺；SCTR 先用图聚类算法按距离把传感器分簇、再按簇置缺；SCTC 按时空块置缺[^src-guo-imputation-evaluation]。

## 与缺失机制分类的关系

本分类刻画缺失的**时空几何结构**；[[missing-not-at-random|Rubin 的 MCAR/MAR/MNAR 分类]]刻画缺失掩码的**统计生成机制**。该论文未将其四类映射到 Rubin 机制术语，两者是不同维度的分类，wiki 中分立记录、不互相归约。

## 对评测结论的影响

在 [[st-traffic-imputation-benchmark|统一评测]]中，缺失模式与数据集特征显著影响排名：评测者的模型选择建议认为时间连续（temporal continuous）缺失模式下 GCASTN 最好、空间连续（spatial continuous）缺失模式下 BRITS 最好（论文未逐一点名对应四类中的哪一个）；IGNNK 在 TW 上多数模式（尤其 TC-）最好但在 SCTR 下表现差（邻近传感器信号同时不可用，LATC 在该模式最好）；SCTC 下依赖复杂时空关系的模型倾向于表现更差[^src-guo-imputation-evaluation]。这构成"插补方法排名必须限定缺失模式"的直接证据：脱离模式谈论插补优劣没有意义。

## 相关页面

- [[st-traffic-imputation-benchmark]] — 使用本分类的统一评测
- [[missing-not-at-random]] — 按统计生成机制的缺失分类（MCAR/MAR/MNAR）
- [[partial-blackout]] — 部分断电式缺失场景（SADI 相关概念）
- [[loft]] — 后续同组论文的实验协议使用 SR-TC/SC-TC 命名的缺失模式（见其页面记录）
- [[pristi]] · [[imputeformer]] · [[std-plm]] — 评测中被比较的方法

[^src-guo-imputation-evaluation]: [[source-guo-imputation-evaluation]]
