---
title: "GSLI: Multi-Scale Graph Structure Learning for Spatial-Temporal Imputation"
type: source-summary
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

# GSLI: Graph Structure Learning for Spatial-Temporal Imputation

**作者**: Xinyu Yang, Yu Sun*, Xinyang Chen*, Ying Zhang, Xiaojie Yuan（南开大学 & 哈尔滨工业大学深圳）[^src-yang-gsli-2025]

**会议**: AAAI 2025

**代码**: https://github.com/GSLI25/GSLI25/

## 核心问题

现有时空填补方法使用固定空间图（给定邻接矩阵）建模空间依赖，隐含假设所有特征的空间关系相同。但论文通过注意力图分析揭示：

1. **特征异质性**（Feature Heterogeneity）：不同特征（如风向 DD vs 风速 FH）在同一对站点间的空间相关性不同。使用统一图会引入误导信息（如用 DBT 站的风速填补 AMS 站的风速会出错，因为机场与市区建筑环境不同）。
2. **跨特征空间依赖**：同一站内不同特征之间存在固定相关性（如 ELD 和 ELL 站的四个特征的注意力图相似），但这种相关性无法由给定图结构反映。

## 核心方法：GSLI

多尺度图结构学习框架，包含四个模块：

### 1. Node-Scale Spatial Learning
- 为每个特征 $f$ 独立学习全局元图 $\dot{G}_f^\Omega$，避免异质特征间的干扰
- 为每个特征分配两对可学习元节点嵌入 $\Omega_1^f, \Omega_2^f \in \mathbb{R}^{N \times d}$
- **显著度建模**（Prominence Modeling）：通过 $P_\Omega^f = \text{MLP}(\Omega_1^f)$ 计算节点显著度向量，用 Hadamard 积增强高影响力节点的边权重
- 元图邻接矩阵：$\dot{A}_\Omega^f = \text{SoftMax}(\text{ReLU}(\tilde{\Omega}_1^f {\Omega_2^f}^\top))$
- 使用图扩散卷积融合给定图和元图的空间依赖
- **Proposition 1**：标准图扩散卷积无法处理特征异质性（证明信息流分析）
- **Proposition 2**：Node-scale 学习可获得期望的特征独立空间依赖

### 2. Feature-Scale Spatial Learning
- 学习元特征图 $\dot{G}^\Phi$，建模所有节点上不同特征间的共同空间相关性
- 可学习元特征嵌入 $\Phi_1, \Phi_2 \in \mathbb{R}^{F \times d}$，同样使用显著度建模
- 将输入按特征维度排列，通过图扩散卷积捕获特征间的空间依赖

### 3. Cross-Feature Representation Learning
- 融合原始表示、node-scale 和 feature-scale 输出：$E = \text{MLP}(\text{Concat}(R \| R^{NL} \| R^{FL}))$
- 使用 Transformer 自注意力捕获不同节点间跨特征的空间依赖
- 时间复杂度 $O(F^2 N T C + F N T C^2)$

### 4. Cross-Temporal Representation Learning
- 在原始输入信号上建模时间依赖（比在填充后信号上更可靠）
- 将输入投影到深度空间后按节点分割，使用 Transformer 自注意力
- 最终输出经 Flatten 和 Concat 获得跨时间表示

## 训练与填补

- 训练时随机选择部分观测值作为标签 $X_B$，其余作为输入 $X_I$
- 损失函数：$L = \mathbb{E}[\|X_B - \hat{X} \odot M_B\|^2]$
- 填补时使用完整缺失数据作为输入，输出 $\hat{X} \odot (1 - M)$

## 实验

### 数据集
6 个真实不完整时空数据集：DutchWind（8688 站, 4 特征）、BeijingMEO（18 站, 5 特征）、LondonAQ（13 站, 3 特征）、CN（140 站, 6 特征）、Los（207 站）、LuohuTaxi（156 站）

### 主要结果
- 在所有缺失率（10%-40%）和所有数据集上取得最优 RMSE 和 MAE
- DutchWind 10% 缺失：RMSE 0.410, MAE 0.205（次优 GRIN: 0.437/0.229）
- LondonAQ 10% 缺失：RMSE 0.272, MAE 0.173（次优 GRIN: 0.311/0.198）
- CN 10% 缺失：RMSE 0.253, MAE 0.120（次优 w/o Prominence: 0.260/0.124）
- 在 MCAR、MAR、MNAR 三种缺失机制下均一致最优

### 消融实验
- **w/o Feature-Split&Scale**：用标准图扩散卷积替代双尺度学习，性能下降
- **w/o Cross-temporal**：移除跨时间表示学习，性能显著下降（LondonAQ RMSE 0.272→0.352）
- **w/o Feature-scale**：仅保留 node-scale，CN RMSE 0.253→0.293
- **w/o Node-scale**：仅保留 feature-scale，CN RMSE 0.253→0.263
- **w/o Prominence**：移除显著度建模，CN RMSE 0.253→0.260
- TemporalGCN 一致低于 TemporalFeatureSA，验证给定邻接矩阵无法准确反映实际空间相关性

## 局限性

- 时间复杂度为 $O(F N^2 T C)$（node-scale），当节点数 N 很大时可能不够高效
- 仅考虑静态图结构学习，未建模动态时空关联
- 未与其他图结构学习方法（如 AGCRN、MTGNN）在填补任务上直接比较

## 关联页面

- [[gsli]] — GSLI 实体页面
- [[node-scale-graph-structure-learning]] — 节点尺度图结构学习
- [[feature-scale-graph-structure-learning]] — 特征尺度图结构学习
- [[prominence-modeling-gsl]] — 显著度建模
- [[imputeformer]] — ImputeFormer 低秩引导 Transformer 填补
- [[cofill]] — CoFILL 条件扩散填补

[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
