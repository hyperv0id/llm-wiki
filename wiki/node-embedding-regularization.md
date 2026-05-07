---
title: "Node Embedding Regularization"
type: concept
tags:
  - regularization
  - graph-neural-network
  - over-parameterization
  - adaptive-graph-learning
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Node Embedding Regularization

节点嵌入正则化指对图神经网络中可学习节点嵌入的过参数化问题进行约束的技术。在自适应图学习中，节点嵌入 $E \in \mathbb{R}^{N \times d}$ 用于捕获节点的空间特征和依赖关系，但往往占模型参数的主要比例，容易导致过拟合[^src-ragc-efficient-traffic-forecasting]。

## 过参数化问题

RAGC 论文分析了 LargeST-CA（8,600 节点）数据集上各模型的参数分布[^src-ragc-efficient-traffic-forecasting]：

| 模型 | 节点嵌入参数占比 | 嵌入使用方式 |
|------|-----------------|-------------|
| GWNet | ~72% | 构建自适应邻接矩阵 |
| AGCRN | ~86% | 构建自适应邻接矩阵 |
| STID | ~67% | 拼接为输入特征 |
| STAEFormer | ~71% | 拼接为输入特征 |
| BigST | ~57% | 拼接为输入特征 |

尽管节点嵌入占参数主导地位，现有方法很少考虑其正则化。

## 正则化方法分类

### 直接作用于嵌入层

| 方法 | 原理 | 局限 |
|------|------|------|
| [[stochastic-shared-embedding\|SSE]] | 随机替换节点嵌入，注入全局平均噪声 | 噪声通过残差传播，需配合抑制机制 |
| Dropout | 随机置零部分嵌入维度 | 破坏空间信息完整性，嵌入维度间的关联被割裂 |
| Laplacian 正则化 | 惩罚预定义图上相邻节点的嵌入距离 | 依赖预定义图结构，缺乏数据驱动灵活性 |

### 间接正则化

| 方法 | 目标 | 局限 |
|------|------|------|
| RGSL | 稀疏化自适应邻接矩阵 | 不直接处理嵌入本身 |
| STC-Dropout | 课程学习 + Dropout | 正则化图信号而非嵌入 |

## RAGC 的协同正则化框架

RAGC 提出 SSE + [[residual-difference-mechanism|RDM]] + 自适应图卷积的三方协同[^src-ragc-efficient-traffic-forecasting]：

1. **SSE** 提供嵌入级正则化，减少对单一嵌入的过度依赖
2. **RDM** 通过减法残差使图卷积权重自适应抑制 SSE 噪声
3. **自适应图卷积** 聚合空间信息，使噪声过滤成为可能

缺少任何一方都会导致性能退化：无 SSE → 过拟合；无 RDM → 训练不稳定；无 AGC → 欠拟合且无法抑制噪声[^src-ragc-efficient-traffic-forecasting]。

## 相关页面

- [[ragc]] — 首次系统解决节点嵌入正则化的交通预测模型
- [[stochastic-shared-embedding|SSE]] — 嵌入层正则化技术
- [[residual-difference-mechanism|RDM]] — 噪声抑制机制
- [[traffic-forecasting]] — 交通预测方法概览

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
