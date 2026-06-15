---
title: "Sandglass Attention"
type: technique
tags:
  - attention
  - spatial-temporal
  - efficiency
  - high-order-correlation
  - plm
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: high
status: active
---

# Sandglass Attention (SGA)

Sandglass Attention（沙漏注意力）是 [[std-plm|STD-PLM]] (AAAI 2025) 提出的高效注意力模块，通过先聚合再恢复的"沙漏"形操作捕获非 pairwise 和高阶时空相关性，同时显著降低基于 PLM 的时空模型的计算开销[^src-std-plm]。

## 设计动机

PLM 的大型嵌入维度下，随着空间 token 数 $N$ 增加（实际应用中 $N$ 通常很大），训练和推理成本急剧上升。此外，节点级空间 token 难以捕获非 pairwise 的高阶时空相关性[^src-std-plm]。

## 结构

### 两阶段操作

| 阶段 | 操作 | 输入→输出 |
|------|------|----------|
| **Precoder** | 可学习查询矩阵 $H_l \in \mathbb{R}^{M \times d_{PLM}}$ 通过 cross-attention 聚合节点级 token | $Z_S \in \mathbb{R}^{N \times d_{PLM}}$ → $Z_H \in \mathbb{R}^{M \times d_{PLM}}$，$M < N$ |
| PLM 处理 | 区域级 token 与时间 token 拼接后送入 PLM | $Z_H, Z_T$ → $Z_H', Z_T'$ |
| **Decoder** | 以恢复的节点级 token 为 query、$H_l$ 为 key、$Z_H'$ 为 value 做 cross-attention | $Z_H' \in \mathbb{R}^{M \times d_{PLM}}$ → $Z_N' \in \mathbb{R}^{N \times d_{PLM}}$ |

### 约束损失

为防止 SGA 的可学习查询矩阵过拟合，设计了约束损失 $L_C = L_G + L_R$[^src-std-plm]：

- **$L_G$（结构感知损失）**：利用邻接矩阵 $A$ 约束注意力权重分布，使 $SGA$ 自动聚焦于更大的完全子图或强连通区域
- **$L_R$（正则化项）**：通过 Dirichlet 分布 $\pi$ 间接控制注意力分布，防止某些节点的注意力权重坍塌为零

## 效率与效果

在 PEMS 数据集上的推理效率[^src-std-plm]：

| 数据集 | 时间（s） | GPU 内存（MiB） |
|--------|----------|-----------------|
| PEMS04 w/ SGA | 7.40 | 8,554 |
| PEMS04 w/o SGA | 17.96 | 15,366 |
| PEMS08 w/ SGA | 9.15 | 15,020 |
| PEMS08 w/o SGA | 52.82 | 29,718 |

SGA 不仅大幅提升效率，还通过区域级 token 的交互捕获了节点级 token 难以表达的高阶时空相关性[^src-std-plm]。

## 与类似方法的区别

SGA 与 SSTBAN (Guo et al., ICDE 2023) 和 CrossFormer (Zhang & Yan, ICLR 2023) 相似——都通过可学习参考点结合注意力机制学习高阶相关性。但现有方法未充分利用图的邻接关系，无法实现不同图间的零样本学习；SGA 通过约束损失将邻接矩阵信息显式注入注意力学习过程[^src-std-plm]。

## Connections

- 论文：[[std-plm|STD-PLM]] — SGA 所属的框架
- 对比：[[cbsa|CBSA]] — Contract-and-Broadcast Self-Attention，另一种压缩-恢复注意力范式
- 对比：[[quest-attention|QUEST Attention]] — 通过 token clustering 做高效长距离注意力
- 概念：[[mixture-of-experts]] — 注意力压缩与 MoE 路由的同构视角

[^src-std-plm]: [[source-std-plm]]
