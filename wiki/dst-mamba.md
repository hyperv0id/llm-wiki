---
title: "DST-Mamba"
type: entity
tags:
  - traffic-prediction
  - mamba
  - state-space-model
  - spatio-temporal
  - decomposition
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# DST-Mamba

**DST-Mamba**（Decomposed Spatio-Temporal Mamba）是一种面向长程交通预测的时空状态空间模型，由 He、Ji 和 Lei（北京工业大学）在 AAAI 2025 上提出。它将时间序列分解与 [[mamba|Mamba]] 选择性 SSM 结合，分别用不同的 backbone 处理季节性和趋势性模式[^src-dst-mamba]。

## 核心设计

DST-Mamba 基于两个关键观察[^src-dst-mamba]：

1. **时空纠缠**：长程交通数据中季节波动与长期趋势相互缠绕，统一建模增加学习难度。现有分解方法（[[autoformer|Autoformer]], [[fedformer|FEDformer]]）仅处理时序纠缠，忽略路网空间约束。
2. **计算效率**：Transformer 在时空场景下的二次复杂度限制了长程/大规模应用。Mamba 的近线性复杂度提供了更高效的替代方案。

### 架构概览

```
输入交通序列 X (L × N)
         │
    ┌────┴────┐
    ▼         ▼
  趋势 X_TR   季节 X_SE
    │         │
    ▼         ▼
 多尺度     双向 Mamba 编码器
 线性预测   (前向 + 后向)
    │         │
    ▼         ▼
   Ŷ_TR      Ŷ_SE
    │         │
    └────┬────┘
         ▼
    Ŷ = Ŷ_SE + λŶ_TR
```

### 季节组件：双向 Mamba 编码器

视图从时间维度切换到**空间维度**（view shift）[^src-dst-mamba]：

1. **Tokenization**：通过图聚合（EI = S · X_SE）将每个节点的完整时间序列压缩为节点 token，整合图结构信息
2. **自适应空间嵌入**：可学习的节点嵌入 EA ∈ R^(N×DA)，保留节点独立特征
3. **双向 Mamba**：前向 Mamba + 后向 Mamba 并行处理，拼接两个方向的输出，避免单向 Mamba 的序列顺序偏差
4. **FFN**：在节点 token 维度上隐式编码时序依赖——因为每个 token 包含完整时间序列，FFN 可自然保留动态顺序关系

### 趋势组件：多尺度线性预测

线性模型擅长捕捉长期稳定趋势[^src-dst-mamba]：

1. **多尺度下采样**：平均池化生成 m 个尺度 {XTR_0, ..., XTR_{m-1}}，从最精细到最宏观
2. **自上而下混合**：从最粗尺度逐级向下混合，去除细节噪声，实现跨尺度信息交互
3. **逐尺度线性预测**：各尺度独立线性映射 → 聚合为综合趋势预测

## 实验结果

在 5 个交通数据集（Traffic, PEMS03/04/07/08）上全面对比 3 类 8 个 baseline[^src-dst-mamba]：

- **Transformers**：iTransformer、Crossformer、PatchTST、FEDformer、Autoformer
- **线性模型**：DLinear
- **Mamba 模型**：[[s-mamba|S-Mamba]]、SOR-Mamba

DST-Mamba 在多数数据集和预测 horizon（12–96 步）上取得 SOTA，计算效率优于 Transformer 基线[^src-dst-mamba]。

### 消融实验（PEMS08）

关键发现[^src-dst-mamba]：

- **分解**（w/o Dec.）：移除后 Avg MSE 从 0.119 升至 0.139（+16.8%），证明分解是基础性设计
- **季节组件**（w/o Sea.）：移除后性能崩溃（Avg MSE 0.513），比移除趋势组件影响更大——空间依赖性对交通预测至关重要
- **双向 Mamba**（w/o Bi-D）：替换为单向后 Avg MSE 升至 0.128，丢失一半空间方向信息
- **自适应嵌入**（w/o AE）：移除后 Avg MSE 升至 0.130，节点独立特征补充了跨节点相关性信息

## 与 S-Mamba 的区别

| 维度 | [[s-mamba|S-Mamba]] (Neurocomputing 2024) | DST-Mamba (AAAI 2025) |
|------|------|------|
| 场景 | 通用 MTSF | 时空交通预测 |
| 核心维度 | 变量间相关性（VC） | 空间节点依赖性 |
| 输入结构 | 多变量时间序列 | 图结构 + 多变量序列 |
| 分解 | 无 | 时序分解（趋势/季节） |
| Mamba 方向 | 双向（变量维度） | 双向（空间节点维度） |
| 趋势建模 | FFN | 多尺度线性预测 |

## 适用场景

- 长程交通预测（horizon ≥ 48 步）
- 大规模路网（节点数 > 100）场景
- 对计算效率有严格要求的实时预测系统

## 局限

- 分解权重 λ 需人工调节（论文中使用 0.5–1.0）[^src-dst-mamba]
- 空间嵌入维度对节点数敏感
- 仅验证于交通领域数据集，未测试通用 MTSF

## 相关页面

- [[mamba|Mamba]] — 选择性 SSM，DST-Mamba 的核心 backbone
- [[s-mamba|S-Mamba]] — 首个 Mamba-based MTSF 框架，DST-Mamba 在时空维度的扩展
- [[spatio-temporal-decomposition]] — 时空分解的概念与对比
- [[multi-scale-linear-prediction]] — 多尺度线性预测技术
- [[traffic-forecasting|交通预测]] — 时空预测的核心应用场景

[^src-dst-mamba]: [[source-dst-mamba]]
