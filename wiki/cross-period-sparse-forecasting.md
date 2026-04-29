---
title: "Cross-Period Sparse Forecasting"
type: technique
tags:
  - sparse-modeling
  - periodicity
  - lightweight
  - downsampling
  - time-series
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Cross-Period Sparse Forecasting

Cross-Period Sparse Forecasting（跨周期稀疏预测）是 SparseTSF 的核心技术，通过将原始序列按周期下采样为子序列，用共享参数的轻量 backbone 预测，从而实现极端的模型压缩[^src-sparsetsf]。

## 核心思想

时序数据通常具有固定周期（如日周期 w=24，周周期 w=168）。传统方法直接建模 L×H 的全连接关系，参数量巨大。Sparse Forecasting 将任务重新表述为：

**跨周期趋势预测**：不直接预测每个时间点，而是预测"同一周期位置"的趋势变化。

## 算法流程

### 1. 下采样（Downsampling）

将原始序列 x ∈ R^L 按周期 w 重排为矩阵 X ∈ R^{w×n}，其中 n = ⌊L/w⌋：

```
X[i,k] = x[i + k×w],  i = 1,...,w,  k = 1,...,n
```

### 2. 预测（Forecasting）

用共享参数的 backbone（Linear 或 MLP）对每个子序列独立预测：

```
Ŷ[i] = f_θ(X[i]),  i = 1,...,w
```

### 3. 上采样（Upsampling）

将预测的子序列 Ŷ ∈ R^{w×m}（m = ⌊H/w⌋）重排回完整预测序列 ŷ ∈ R^H：

```
ŷ[i + l×w] = Ŷ[i,l],  i = 1,...,w,  l = 1,...,m
```

### 4. 滑动聚合（可选）

为缓解下采样带来的信息损失，使用 1D 卷积进行滑动聚合：

```
x' = x + Conv1D(x)  // kernel size = 2×⌊w/2⌋+1
```

## 参数量分析

对于 Linear backbone：
- Conv1D 参数：2×⌊w/2⌋+1
- Linear 参数：n × m = ⌊L/w⌋ × ⌊H/w⌋
- **总计**：⌊L/w⌋ × ⌊H/w⌋ + 2×⌊w/2⌋+1

当 L=720, H=720, w=24 时，参数量仅为 **925**，相比全连接 Linear 的 518,400 参数，压缩了 **560 倍**。

## 隐式正则化

论文证明 Sparse 技术等价于在原始权重矩阵 W_f ∈ R^{H×L} 上施加两类正则化：

1. **结构稀疏正则** α‖W_forbid‖₁：惩罚非对角块元素，强制周期间参数为 0
2. **权重共享正则** β‖σ²(W_shared)‖₁：约束对角块内元素方差，强制周期内参数一致

这相当于隐式 L1 正则化，使模型更鲁棒。

## 与其他技术的对比

| 技术 | 周期利用方式 | 参数量级 |
|------|-------------|----------|
| DLinear | 无周期建模 | ~1k |
| FITS | 频域低通滤波 | ~10k |
| CycleNet | 可学习周期向量 | ~10k |
| **Cross-Period Sparse** | **跨周期下采样** | **<1k** |

## 超参数

- **w (周期长度)**：应与数据的内在周期对齐
  - ETTh1/ETTh2: w=24（日周期）
  - Electricity: w=168（周周期）
  - Traffic: w=168（周周期）
  - Weather: w=4（数据采样间隔 10min，日周期 144 步太长，用较小值）

## 引用

[^src-sparsetsf]: [[source-sparsetsf]]