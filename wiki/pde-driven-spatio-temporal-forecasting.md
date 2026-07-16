---
title: "PDE-Driven Spatio-Temporal Forecasting"
type: concept
tags:
  - pde
  - spatio-temporal-forecasting
  - physics-inspired
  - neural-pde
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# PDE-Driven Spatio-Temporal Forecasting

PDE-driven spatio-temporal forecasting 是一类将时空预测重新形式化为**偏微分方程演化**的方法，区别于传统上将时空动态视为离散图推理的主流范式[^src-stpde]。

## 核心思想

将时空观测视为来自底层连续场 $u(x, \tau)$ 的样本，其动力学由 PDE 支配：

$$\frac{\partial u}{\partial \tau} = \mathcal{F}(u, \nabla u, \nabla^2 u, \dots; \theta(x))$$

其中 $\mathcal{F}$ 包含域不变算子（如拉普拉斯 $\nabla^2$）和空间变化的系数 $\theta(x)$（编码异质性）[^src-stpde]。

## 与主流 STGNN 的关键差异

| 维度 | 离散图推理（STGNN） | PDE 驱动（STPDE 类） |
|------|---------------------|----------------------|
| 空间建模 | 图邻接矩阵 + 消息传递 | 连续微分算子 + 核积分 |
| 异质性 | 隐式（学到的嵌入） | 显式（空间变化系数） |
| 泛化 | 依赖图结构先验 | 物理先验支持域迁移 |
| 长程耦合 | 需要多层堆叠 | 全局核积分（单层） |

## 实例

- **[[stpde|STPDE]]**（Liu et al., ICML 2026）：非齐次扩散方程 + Green 函数注意力 + Environment Basis Manifold，覆盖 ID/OOD/迁移/持续学习四场景
- **[[ctenet|CTENet]]**（NeurIPS 2025）：ADR（对流-扩散-反应）方程嵌入神经网络架构，用于空气质量预测
- **[[dyffusion|DYffusion]]**（NeurIPS 2023）：将扩散模型解释为动力系统演化过程，用于时空预测
- **[[source-pi-mfm|PI-MFM]]**（arXiv 2025）：物理信息多模态基础模型，PDE 编码的 Method of Lines + 物理损失预训练

## 优势与局限

**优势**：
- 物理先验提供强归纳偏置，天然支持 OOD 泛化
- 全局 Green 函数视角无需多层消息传递即可实现长程耦合
- 系数分解使迁移学习中"冻结物理 + 适应环境"成为可能

**局限**[^src-stpde]：
- PDE 形式选择（如是否只用扩散、是否加对流项）依赖领域知识
- 在高度非物理场景（如极端事件）下的适用性有待验证
- 计算上核积分需精心设计以避免 $O(N^2)$

[^src-stpde]: [[source-stpde]]
