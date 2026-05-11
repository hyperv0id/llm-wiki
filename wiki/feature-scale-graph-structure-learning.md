---
title: "Feature-Scale Graph Structure Learning"
type: technique
tags:
  - graph-structure-learning
  - spatio-temporal
  - data-imputation
  - cross-feature-dependency
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: medium
status: active
---

# Feature-Scale Graph Structure Learning

Feature-Scale Graph Structure Learning 是 [[gsli|GSLI]] 的核心模块之一，通过学习元特征图来建模所有节点上不同特征间的共同空间相关性，弥补给定图结构无法反映特征间相关性的缺陷[^src-yang-gsli-2025]。

## 核心思想

论文通过注意力图可视化发现（Figure 1d）：同一站点内不同特征的注意力图相似，说明特征间存在固定的空间相关性。但给定的图结构仅连接站点（节点），无法反映特征间的这种相关性。Feature-scale 学习通过构建元特征图 $\dot{G}^\Phi$ 捕获这种跨特征的空间依赖[^src-yang-gsli-2025]。

## 方法

### 元特征嵌入与显著度建模

定义两对可学习元特征嵌入：
- 源特征嵌入 $\Phi_1 \in \mathbb{R}^{F \times d}$
- 目标特征嵌入 $\Phi_2 \in \mathbb{R}^{F \times d}$

通过显著度建模增强对填补贡献更大的特征的权重：

$$P_\Phi = \text{MLP}(\Phi_1)$$
$$\tilde{\Phi}_1 = \Phi_1 \odot P_\Phi$$

异质性较低、对其他特征填补贡献更大的特征将获得更强的源嵌入权重[^src-yang-gsli-2025]。

### 元特征图构建

$$\dot{A}^\Phi = \text{SoftMax}(\text{ReLU}(\tilde{\Phi}_1 \Phi_2^\top))$$

该邻接矩阵 $\dot{A}^\Phi \in \mathbb{R}^{F \times F}$ 表示特征间的共同空间相关性。

### 图扩散卷积

将输入按特征维度排列 $R' = \text{Permute}_{(2,0,1,3)}(R) \in \mathbb{R}^{F \times N \times T \times C}$，然后应用图扩散卷积：

$$R'^{FL} = \sum_{k=0}^{K} (\dot{A}^\Phi)^k R' \Theta_k^\Phi$$

输出按节点维度重新排列回 $R^{FL} = \text{Permute}_{(1,2,0,3)}(R'^{FL}) \in \mathbb{R}^{N \times T \times F \times C}$[^src-yang-gsli-2025]。

### 与 Node-Scale 的互补

- **Node-scale**：为每个特征独立学习全局空间依赖，解决特征异质性
- **Feature-scale**：学习特征间的共同空间相关性，捕获给定图无法反映的跨特征依赖

两者互补：Node-scale 避免误导，Feature-scale 挖掘额外信息。

## 复杂度

- 时间：$O(F^2 N T C + F N T C^2 + F^2 d + F d^2)$
- 空间：$O(F N T C + C^2 + F^2 + F d + d^2)$

## 消融证据

移除 feature-scale 学习（仅保留 node-scale）后，CN 数据集 RMSE 从 0.253 降至 0.293（+15.8%），LondonAQ RMSE 从 0.272 降至 0.308（+13.2%）。Feature-scale 的移除影响大于 node-scale，表明跨特征空间依赖对填补性能至关重要[^src-yang-gsli-2025]。

## 关联页面

- [[gsli]] — GSLI 框架
- [[node-scale-graph-structure-learning]] — 节点尺度图结构学习
- [[prominence-modeling-gsl]] — 显著度建模
- [[imputeformer]] — ImputeFormer（使用嵌入注意力隐式建模空间关系）
- [[fourier-imputation-loss]] — 傅里叶填补损失（频域正则化方法）

[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
