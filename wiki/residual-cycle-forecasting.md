---
title: "Residual Cycle Forecasting (RCF)"
type: technique
tags:
  - time-series-forecasting
  - periodicity-modeling
  - decomposition
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Residual Cycle Forecasting (RCF)

RCF 是 CycleNet 的核心技术，由华南理工大学等机构提出（NeurIPS 2024），通过显式建模时序数据的周期性模式来提升长期预测性能[^src-cyclenet]。

## 核心思想

RCF 本质上是一种季节性-趋势分解（STD）方法，其核心思想分为两步：

1. **周期建模**：使用可学习的循环周期 $Q \in \mathbb{R}^{W \times D}$ 显式建模数据内在的周期性模式[^src-cyclenet]
2. **残差预测**：从原始输入中移除周期分量，对残差分量进行预测，最后加回周期分量得到最终预测[^src-cyclenet]

## 技术细节

### 可学习循环周期

给定通道数 $D$ 和先验周期长度 $W$，生成可学习循环周期 $Q \in \mathbb{R}^{W \times D}$，初始化为零[^src-cyclenet]。这些循环周期在通道内全局共享，通过循环复制得到与输入序列等长的周期分量 $C$[^src-cyclenet]。

### 周期分量对齐

由于循环周期 $Q$ 是虚拟序列，需要通过对齐和复制操作获取等效的周期分量：

- 对历史输入 $x_{t-L+1:t}$：左移 $Q$ $t \mod W$ 位，然后重复 $\lfloor L/W \rfloor$ 次[^src-cyclenet]
- 对未来预测 $x_{t+1:t+H}$：左移 $Q$ $(t+L) \mod W$ 位，然后重复 $\lfloor H/W \rfloor$ 次[^src-cyclenet]

### 残差预测流程

1. 从原始输入中移除周期分量得到残差：$x'_{t-L+1:t} = x_{t-L+1:t} - c_{t-L+1:t}$[^src-cyclenet]
2. 将残差通过骨干网络（Linear 或 MLP）得到残差预测：$\bar{x}'_{t+1:t+H}$[^src-cyclenet]
3. 将残差预测加回周期分量得到最终预测：$\bar{x}_{t+1:t+H} = \bar{x}'_{t+1:t+H} + c_{t+1:t+H}$[^src-cyclenet]

## 即插即用特性

RCF 可作为即插即用模块，显著提升现有模型的预测精度。实验表明，RCF 可将 PatchTST 和 iTransformer 的预测精度显著提升[^src-cyclenet]。

## 与其他 STD 方法的比较

| 方法 | 周期建模方式 |
|------|-------------|
| Autoformer/FEDformer | 移动平均分解 |
| DLinear | 季节性-趋势分离 |
| DEPTS | 参数化周期函数 |
| SparseTSF | 跨周期稀疏预测 |
| **RCF** | **可学习循环周期** |

## 优势

- 概念简单、计算高效[^src-cyclenet]
- 可作为即插即用模块提升现有模型[^src-cyclenet]
- 只需额外 $W \times D$ 个可学习参数，无额外 MACs[^src-cyclenet]

## 局限性

- 在具有显著极端值的数据集上性能可能受影响[^src-cyclenet]
- 当前版本仅考虑单通道关系建模[^src-cyclenet]

## 引用

[^src-cyclenet]: [[source-cyclenet]]