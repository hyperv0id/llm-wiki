---
title: "Graph Learning Layer"
type: technique
tags:
  - graph-structure-learning
  - graph-neural-network
  - spatial-dependency
  - adaptive
created: 2026-05-30
last_updated: 2026-05-30
source_count: 2
confidence: medium
status: active
---

# Graph Learning Layer

Graph Learning Layer（图学习层）是 [[mtgnn|MTGNN]] (KDD 2020) 提出的自适应图结构学习模块，在无需预定义邻接矩阵的情况下从多变量时间序列数据中学习变量间的单向依赖关系 [^src-mtgnn]。

## 动机

多变量时间序列的变量间依赖关系通常是未知且非对称的——道路 A 的交通状况可能影响道路 B，但反过来未必成立。传统方法使用对称距离度量（点积、欧氏距离）构建图结构，无法捕捉这种单向性 [^src-mtgnn]。

## 公式化表达

给定两组可学习节点嵌入 E₁, E₂ ∈ ℝ^(N×d)（随机初始化，训练中学习），图学习层计算：

M₁ = tanh(α·E₁·Θ₁)
M₂ = tanh(α·E₂·Θ₂)
A = ReLU(tanh(α·(M₁·M₂ᵀ − M₂·M₁ᵀ)))

其中 Θ₁, Θ₂ 为可学习参数矩阵，α 为控制 tanh 饱和速率的超参数 [^src-mtgnn]。

### 单向性保证

M₁M₂ᵀ − M₂M₁ᵀ 的减法 + ReLU 激活确保如果 A_uv > 0，则对角线对应项 A_vu = 0 [^src-mtgnn]。即任何节点对之间只保留单向边——这符合交通流等场景中因果方向性的直觉。

### 稀疏化

对每个节点，仅保留 top-k 最大权重的邻居（argtopk），其余边权重置零 [^src-mtgnn]。既降低后续图卷积成本，又防止全连接邻接矩阵引入噪声。

### 融合外部知识

当需要融入静态节点属性（如道路特征）时，可设置 E₁ = E₂ = Z，其中 Z 为静态节点特征矩阵。MTGNN 选择学习静态图而非动态图，因为动态图（每个时间步重新计算邻接矩阵）在同时需要学习图结构时使模型极难收敛 [^src-mtgnn]。

## 训练与推理

- **训练**：通过子图采样——每轮迭代随机分组节点，图学习层仅计算组内节点对相似度，复杂度从 O(N²) 降至 O((N/s)²) [^src-mtgnn]
- **推理**：所有节点嵌入已训练完毕，可预计算全局邻接矩阵。虽仍为 O(N²)，但可在推理前并行完成 [^src-mtgnn]

## 图学习方法比较

在 METR-LA 上的消融实验 [^src-mtgnn]：

| 方法 | MAE | RMSE | MAPE |
|------|-----|------|------|
| 预定义 A（道路距离） | 2.9017 | 6.1288 | 0.0836 |
| 无向 A (M₁M₁ᵀ) | 2.7736 | 5.8411 | 0.0783 |
| 有向 A (M₁M₂ᵀ) | 2.7758 | 5.8217 | 0.0783 |
| 动态 A (逐时间步) | 2.8124 | 5.9189 | 0.0794 |
| **单向 A（本文）** | **2.7715** | **5.8070** | **0.0778** |

单向 A 在 RMSE 上优于所有变体，且比预定义 A 的 RMSE 降低 5.2% [^src-mtgnn]。

## 与其他图学习方法的区别

MTGNN 的图学习服务于*预测*任务，学习到的邻接矩阵是稳定的（在训练集时间范围内共享）。这与 [[gsli|GSLI]]（AAAI 2025）的图学习不同——后者面向*填补*任务，需为不同节点和特征尺度学习独立的图结构 [^src-yang-gsli-2025]。

## 相关页面

- [[mtgnn]] — 使用图学习层的 MTGNN 模型
- [[mix-hop-propagation-layer]] — 下游图卷积模块
- [[cross-dimension-dependency]] — 图学习层建模的核心目标
- [[gsli]] — 后续图结构学习方法（填补任务）

[^src-mtgnn]: [[source-mtgnn]]
[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
