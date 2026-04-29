---
title: "SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling"
type: source-summary
tags:
  - time-series
  - forecasting
  - lightweight
  - sparse-modeling
  - periodicity
  - tpami-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# SparseTSF

## 论文信息

- **标题**: SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling
- **作者**: Shengsheng Lin, Weiwei Lin (华南理工大学), Wentai Wu (暨南大学), Haojun Chen, C. L. Philip Chen
- **发表**: **IEEE TPAMI 2026** (Pattern Analysis and Machine Intelligence)
- **会议版本**: ICML 2024 Oral
- **代码**: https://github.com/lss-1138/SparseTSF

## 核心贡献

### Cross-Period Sparse Forecasting 技术

SparseTSF 的核心创新是 **Cross-Period Sparse Forecasting（跨周期稀疏预测）技术**：

1. **下采样**：将原始序列按周期 w 下采样为 w 个子序列
2. **共享参数预测**：用单一 backbone（Linear 或 MLP）对所有子序列进行预测
3. **上采样**：将预测的子序列上采样回完整预测序列

这种设计将原始预测任务简化为**跨周期趋势预测**，实现两个关键效果：
- **参数压缩**：参数量从 O(L×H) 降至 O(L/w × H/w)
- **隐式正则化**：稀疏结构起到类似 L1 正则化的作用，增强鲁棒性

### 极简参数规模

- **SparseTSF/Linear**: < 1,000 参数（首次将 LTSF 模型降至 1k 级别）
- **SparseTSF/MLP**: < 8,000 参数

相比主流模型（百万级参数），SparseTSF 参数量低 1~4 个数量级。

## 理论分析

论文证明 Sparse 技术等价于在原始权重矩阵上施加两类正则化：
1. **结构稀疏**：惩罚非对角块元素，强制稀疏结构
2. **权重共享**：约束对角块内元素方差，实现参数复用

这相当于隐式 L1 正则化，提升模型对噪声的鲁棒性和泛化能力。

## 实验结果

- 在 6 个基准数据集（ETTh1, ETTh2, Electricity, Traffic, Solar-Energy, Weather）上取得 SOTA 或接近 SOTA
- 多变量预测：SparseTSF/MLP 获得 39 次 Top-3（24 个预测任务）
- 单变量预测：SparseTSF/Linear 获得整体 SOTA

## 与本 Wiki 已收录论文的关系

- **与 CycleNet 的关联**：两者都利用周期先验进行轻量化设计，但 CycleNet 使用可学习周期参数，SparseTSF 使用跨周期稀疏预测
- **与 TQNet 的关联**：同属林伟伟团队（华南理工大学），都发表于 2024-2025 年，都关注轻量化建模
- **与 FITS 的对比**：FITS 是首个将模型参数量降至 10k 级别的里程碑工作，SparseTSF 将其进一步降至 1k 级别

## 局限性

- 需要预先知道数据周期 w
- 对于弱周期数据，下采样可能导致信息损失
- 高维多变量场景下 MLP 版本参数量仍高于 Linear 版本

[^src-sparsetsf]: [[source-sparsetsf]]