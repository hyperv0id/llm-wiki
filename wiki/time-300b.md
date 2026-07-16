---
title: "Time-300B"
type: entity
tags:
  - time-series
  - dataset
  - pretraining
created: 2026-07-25
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# Time-300B

Time-300B 是 [[time-moe|Time-MoE]] 论文同期发布的大规模时间序列预训练数据集，包含超过 309B 时间点、覆盖 9 个以上领域，是当时最大的开放访问时间序列数据集合[^src-time-moe]。

## 数据组成

| 领域 | 时间点数 | 占比 |
|------|---------|------|
| 自然（天气/气候/空气质量） | 279.7B | 90.5% |
| 能源 | 16.0B | 5.2% |
| 合成数据 | 9.2B | 3.0% |
| 交通 | 2.1B | 0.7% |
| 网络 | 1.8B | 0.6% |
| 其他 | <0.1% | - |

自然领域占据主导，主要来自 Weatherbench（逐小时/日/周）、ERA5、CMIP6 等大规模气象气候数据集[^src-time-moe]。

## 数据清洗管线

Time-MoE 设计了专用的时间序列数据清洗流程[^src-time-moe]：

1. **缺失值处理**：在缺失值（nan/inf）处切分序列为多个子序列，移除缺失段同时保留原始模式完整性——而非用均值填充。
2. **无效观测过滤**：用固定窗口扫描，计算一阶/二阶差分的零值比，超过阈值（0.2）则丢弃窗口。过滤常值填充导致的伪模式。
3. **下采样**：对 Weatherbench、CMIP6、ERA5 等超大气候数据集进行下采样以防止数据不平衡和同质化。
4. **二进制存储**：每个数据集按多文件拆分 + 元信息文件管理，支持固定内存训练加载。

## 与同类数据集的对比

| 数据集 | 时间点 | 发布方 |
|--------|--------|--------|
| **Time-300B** | **309B** | Time-MoE (ICLR 2025) |
| Moirai 数据集 | 27B/231B | Salesforce |
| LOTSA (Chronos) | 84B | Amazon |
| TimesFM | 100B | Google |

Time-300B 比同时期最大的时序预训练数据集大 3~10 倍，且开源了完整的数据处理管线代码[^src-time-moe]。

[^src-time-moe]: [[source-time-moe]]
