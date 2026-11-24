---
title: "CycleNet"
type: entity
tags:
  - time-series-forecasting
  - periodicity-modeling
  - neurips-2024
created: 2026-04-28
last_updated: 2026-11-24
source_count: 4
confidence: medium
status: active
---

# CycleNet

CycleNet 是华南理工大学等机构提出的长期时序预测模型（NeurIPS 2024），通过显式建模周期性模式实现高效准确的预测[^src-cyclenet]。

## 模型架构

CycleNet 由两部分组成：

1. **Residual Cycle Forecasting (RCF)**：核心创新技术，使用可学习的循环周期 $Q \in \mathbb{R}^{W \times D}$ 建模数据内在周期[^src-cyclenet]
2. **骨干网络**：单层 Linear（CycleNet/Linear）或双层 MLP（CycleNet/MLP）[^src-cyclenet]

### RCF 技术流程

1. **周期建模**：生成可学习循环周期 $Q$，按绝对相位 $t\bmod W$ 滚动并循环复制得到周期分量 $C$（查表，不是检索）[^src-cyclenet]
2. **残差预测**：从原始输入中移除周期分量，对残差进行预测，最后加回周期分量[^src-cyclenet]

## 性能表现

| 数据集 | CycleNet/MLP 排名 |
|--------|-------------------|
| ETTh1/2 | 第一 |
| ETTm1/2 | 第一 |
| Electricity | 第一 |
| Solar-Energy | 第一 |
| Weather | 第一 |
| Traffic | 第二（仅次于 iTransformer）|

## 效率优势

相比 iTransformer，CycleNet/MLP 参数减少超过 10 倍（472.9K vs 5.15M），MACs 减少超过 12 倍[^src-cyclenet]。

## 即插即用

RCF 技术可作为即插即用模块，显著提升现有模型（如 PatchTST、iTransformer）的预测精度[^src-cyclenet]。

## 与其他模型的关系

- **与 DLinear**：CycleNet/Linear 与 DLinear 使用相同的单层线性骨干，但 RCF 技术带来显著性能提升[^src-cyclenet]
- **与 Autoformer/FEDformer**：RCF 本质上是一种季节性-趋势分解方法，但显式建模全局周期模式[^src-cyclenet]
- **与 SparseTSF**：两者都利用周期信息，但 RCF 使用可学习周期而非跨周期稀疏预测[^src-cyclenet]
- **与 [[source-timebase|TimeBase]]（同槽位对照）**：两者均走轻量周期线性预测路线，但 CycleNet 维护按绝对相位 $t\bmod W$ 查表滚转的全局可学习周期表 $Q$ 做残差剥离与加回，TimeBase 则不做相位查表，而是按周期 $P$ 切段后经正交正则线性层提取 $R=6$ 个时间基直接重构未来分段（参数降至 $0.39\text{K}$ 量级）；两者均依赖周期先验且采用通道独立假设[^src-timebase]
- **与 [[source-dsr-tsf|DSR-TSF]]（跨槽位正交）**：CycleNet 维护按绝对相位 $t\bmod W$ 查表滚转的全局可学习周期表 $Q$ 做显式周期残差剥离与加回，前提是全局周期结构稳定；DSR-TSF 则不维护全局周期表，而是在单窗内用 `rFFT` 提取主导频谱特征，软路由 $R$ 条隐式时域响应分支对输入做自适应残差修正后再经分段低秩头预测，针对同一变量跨窗口主导周期漂移的场景[^src-dsr-tsf]
- **与 [[source-mixlinear|MixLinear]]（同槽位互斥）**：两者均面向轻量周期长时预测，但 CycleNet 维护按绝对相位滚转的全局可学习周期表 $Q$ 做残差剥离与加回，MixLinear 则对下采样序列并联时域 2D 分段双线性与频域秩-2 复投影双支相加（45–176 参数），无需全局周期表查表；两者缺 MAE 同口径及部分同代基线对照[^src-mixlinear]

## 局限性

- 在具有显著时空特性和极端值的 Traffic 数据集上性能略逊于 iTransformer[^src-cyclenet]
- 当前版本仅考虑单通道关系建模[^src-cyclenet]

## 引用

[^src-cyclenet]: [[source-cyclenet]]
[^src-timebase]: [[source-timebase]]
[^src-dsr-tsf]: [[source-dsr-tsf]]
[^src-mixlinear]: [[source-mixlinear]]