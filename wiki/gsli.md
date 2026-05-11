---
title: "GSLI"
type: entity
tags:
  - graph-structure-learning
  - spatio-temporal
  - data-imputation
  - feature-heterogeneity
  - aaai-2025
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: medium
status: active
---

# GSLI

**GSLI**（Multi-Scale Graph Structure Learning for Spatial-Temporal Imputation）是南开大学和哈尔滨工业大学深圳于 AAAI 2025 提出的时空数据填补框架，核心思想是**多尺度图结构学习**以适应特征异质性和跨特征空间依赖[^src-yang-gsli-2025]。

## 核心动机

现有方法使用固定空间图建模空间依赖，隐含假设所有特征的空间关系相同。但现实时空数据中：

1. **特征异质性**：不同特征（如风向 vs 风速）在同一对站点间的空间相关性不同，统一图会引入误导
2. **跨特征空间依赖**：同一站内不同特征间存在固定相关性，但给定图无法反映

论文通过注意力图可视化验证了这两个观察（Figure 1c/1d）[^src-yang-gsli-2025]。

## 架构

GSLI 由四个模块组成：

```
输入 X ∈ R^{N×T×F}
    │
    ├──► Node-Scale Spatial Learning（每个特征独立学习元图）
    │     ├─ 元节点嵌入 Ω₁ᶠ, Ω₂ᶠ ∈ R^{N×d}
    │     ├─ 显著度建模 P_Ω^f = MLP(Ω₁ᶠ)
    │     └─ 图扩散卷积（给定图 + 元图）
    │
    ├──► Feature-Scale Spatial Learning（学习特征间空间相关性）
    │     ├─ 元特征嵌入 Φ₁, Φ₂ ∈ R^{F×d}
    │     ├─ 显著度建模 P_Φ = MLP(Φ₁)
    │     └─ 图扩散卷积（元特征图）
    │
    ├──► Cross-Feature Representation Learning
    │     ├─ 融合 R ∥ R^NL ∥ R^FL
    │     └─ Transformer 自注意力（跨节点跨特征）
    │
    └──► Cross-Temporal Representation Learning
          └─ Transformer 自注意力（跨时间步）
```

## 关键创新

### Node-Scale 图结构学习
为每个特征 $f$ 独立学习元图 $\dot{G}_f^\Omega$，避免异质特征间的干扰。通过 Proposition 1 证明标准图卷积无法处理特征异质性，Proposition 2 证明 node-scale 学习可获得期望的特征独立空间依赖[^src-yang-gsli-2025]。

### Feature-Scale 图结构学习
学习元特征图 $\dot{G}^\Phi$，建模同一节点内不同特征间的共同空间相关性。这弥补了给定图结构无法反映特征间相关性的缺陷[^src-yang-gsli-2025]。

### 显著度建模
通过 $P_\Omega^f = \text{MLP}(\Omega_1^f)$ 计算节点/特征显著度向量，用 Hadamard 积增强高影响力节点的边权重。消融实验显示显著度建模在 CN 数据集上贡献 MAE 0.120→0.124[^src-yang-gsli-2025]。

## 实验结果

| 数据集 | 缺失率 | GSLI RMSE | 次优 RMSE | 提升 |
|--------|--------|-----------|-----------|------|
| DutchWind | 10% | 0.410 | 0.437 (GRIN) | 6.2% |
| BeijingMEO | 10% | 0.399 | 0.402 (w/o Prom) | 0.7% |
| LondonAQ | 10% | 0.272 | 0.311 (GRIN) | 12.5% |
| CN | 10% | 0.253 | 0.260 (w/o Prom) | 2.7% |
| Los | 10% | 0.263 | 0.295 (PriSTI) | 10.8% |
| LuohuTaxi | 10% | 0.410 | 0.436 (PoGeVon) | 6.0% |

在 MCAR、MAR、MNAR 三种缺失机制下均一致最优[^src-yang-gsli-2025]。

## 与相关方法的对比

| 方法 | 图结构 | 特征异质性 | 跨特征依赖 | 显著度建模 |
|------|--------|-----------|-----------|-----------|
| GRIN | 固定图 | ❌ | ❌ | ❌ |
| PriSTI | 学习图（粗粒度） | ❌ | ❌ | ❌ |
| ImputeFormer | 无图（节点嵌入） | ❌ | ❌ | ❌ |
| **GSLI** | **双尺度学习** | **✅** | **✅** | **✅** |

## Connections

- [[node-scale-graph-structure-learning]] — 节点尺度图结构学习
- [[feature-scale-graph-structure-learning]] — 特征尺度图结构学习
- [[prominence-modeling-gsl]] — 显著度建模
- [[imputeformer]] — ImputeFormer，低秩引导 Transformer 填补（同时期工作）
- [[cofill]] — CoFILL，条件扩散填补
- [[traffic-forecasting]] — 交通预测
- [[spatio-temporal-foundation-model]] — 时空基础模型
- [[projected-attention]] — ImputeFormer 的时间投影注意力
- [[embedded-attention]] — ImputeFormer 的空间嵌入注意力

[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
