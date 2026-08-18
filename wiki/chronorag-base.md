---
title: "ChronoRAG Base"
type: concept
tags:
  - time-series-database
  - retrieval-augmented-generation
  - hierarchical-indexing
  - zero-shot-forecasting
  - www-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# ChronoRAG Base (CRB)

**ChronoRAG Base** 是 [[quizsf|QuiZSF]] 框架中设计的大规模层次化时序数据库，用于支持检索增强的零样本时序预测[^src-quizsf-zero-shot-forecasting-www26]。其核心设计目标是实现高效、域敏感的时序存储与检索，使 RAG 范式从文本扩展到时间序列领域[^src-quizsf-zero-shot-forecasting-www26]。

## 数据构成

CRB 整合 27 个时序数据集，覆盖 7 个领域：Web、Energy、Health、IoT、Nature、Transport、Environment[^src-quizsf-zero-shot-forecasting-www26]。数据来源包括 UTSD、TSER Archive、Monash、TDBrain 和 UCR Time Series Archive[^src-quizsf-zero-shot-forecasting-www26]。采样频率从日级到毫秒级（如 TDBrain 0.002 秒）[^src-quizsf-zero-shot-forecasting-www26]。

三个版本[^src-quizsf-zero-shot-forecasting-www26]：

| 版本 | Time Points |
|------|-------------|
| CRB-Small | 34M |
| CRB-Medium | 48M |
| CRB-Large | 143M |

CRB-Medium 为实验默认选择[^src-quizsf-zero-shot-forecasting-www26]。

## 数据协议

统一数据处理协议包括[^src-quizsf-zero-shot-forecasting-www26]：

- **滑窗分段**：窗口大小 w，步长 s，保留局部模式并提升检索效率[^src-quizsf-zero-shot-forecasting-www26]。
- **线性插值**：填充缺失值以保证数据完整性[^src-quizsf-zero-shot-forecasting-www26]。
- **通道独立处理**：每个维度独立处理（遵循 PatchTST/Time-LLM/TTM 等验证的范式）[^src-quizsf-zero-shot-forecasting-www26]。
- **统一元数据**：标准化 item ID、起止时间、频率、域、序列值等属性[^src-quizsf-zero-shot-forecasting-www26]。
- **ARROW 格式存储**：优化深度学习框架访问效率[^src-quizsf-zero-shot-forecasting-www26]。

## Hierarchical Series Tree

层次化树结构用于高效索引和检索[^src-quizsf-zero-shot-forecasting-www26]：

1. 顶层按域分区，将数据集划分为 K 个不重叠域组[^src-quizsf-zero-shot-forecasting-www26]。
2. 每个域组内用 k-means 递归聚类，每个簇最多 256 条序列（参考 Marigold 聚类粒度）[^src-quizsf-zero-shot-forecasting-www26]。
3. 每个簇的原型（prototype）选为最接近质心的序列[^src-quizsf-zero-shot-forecasting-www26]。
4. 检索时先匹配簇原型，再仅检查少量候选簇，大幅降低计算量[^src-quizsf-zero-shot-forecasting-www26]。
5. 支持动态更新：新序列匹配最近原型插入对应簇，簇溢出时触发局部重聚类[^src-quizsf-zero-shot-forecasting-www26]。

## Hybrid and Hierarchical Retrieval (HHTR)

HHTR 结合域内局部原型匹配与全局原型比较[^src-quizsf-zero-shot-forecasting-www26]：

$$\text{Top-K} = \rho \cdot \text{Top-K}_{\text{local}} + (1-\rho) \cdot \text{Top-K}_{\text{global}}$$

当目标域未知时退化为纯全局检索[^src-quizsf-zero-shot-forecasting-www26]。距离度量采用余弦相似度与欧氏距离逆数的复合分数[^src-quizsf-zero-shot-forecasting-www26]。

## 关联页面

- [[quizsf]] — QuiZSF 检索增强零样本预测框架
- [[source-quizsf-zero-shot-forecasting-www26]] — 源文件摘要
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF 范式
- [[gtr]] — 全局时序检索模块
- [[ratd]] — 检索增强时序扩散模型

[^src-quizsf-zero-shot-forecasting-www26]: [[source-quizsf-zero-shot-forecasting-www26]]
