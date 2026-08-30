---
title: "STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model"
type: source-summary
tags:
  - mamba
  - state-space-model
  - spatiotemporal
  - graph
  - 2024
created: 2026-07-07
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model

**Authors**: Lincan Li, Hanchen Wang, Wenjie Zhang, Adelle Coster (University of New South Wales, Sydney)

**Venue**: arXiv 2024 | **arXiv**: 2403.12418 | **Code**: [github.com/LincanLi98/STG-Mamba](https://github.com/LincanLi98/STG-Mamba)

## 核心贡献

STG-Mamba 是 **首个将选择性状态空间模型（Selective SSM / Mamba）应用于时空图（STG）预测** 的工作。[^src-stg-mamba] 它将 STG 网络视为一个动态系统，通过状态空间建模框架统一描述图结构的动态演化，克服了 Transformer 在 STG 任务上二次计算复杂度的瓶颈。

## 关键设计

### GS3B：图选择性状态空间块（Graph Selective State Space Block）

STG-Mamba 采用 Encoder-Decoder 架构，核心模块为 GS3B。每个 GS3B 包含：[^src-stg-mamba]

1. **输入相关的边构建（Input-Dependent Edge Construction）**：根据当前输入的节点特征动态构建图邻接关系，而非依赖固定的预定义邻接矩阵。
2. **动态节点特征选择（Dynamic Node Feature Selection）**：利用 Mamba 的选择机制，控制哪些输入特征流入隐藏状态，实现数据驱动的上下文建模。
3. **状态演化建模**：通过 SSSM 的一阶差分/微分方程描述 STG 系统在时间轴上的状态转移过程。

### KFGN：卡尔曼滤波图神经网络（Kalman Filtering Graph Neural Network）

KFGN 是 STG-Mamba 的核心创新之一，将经典控制理论中的 **卡尔曼滤波（Kalman Filtering）** 引入 GNN 的图结构更新：[^src-stg-mamba]

- **自适应图结构升级**：KFGN 在 SSSM 的上下文框架下，通过可学习的卡尔曼滤波过程动态整合和升级来自不同时间粒度的 STG 嵌入。
- **线性复杂度**：KFGN 在保持线性复杂度 $O(n)$ 的同时，确保图结构与 STG 系统当前状态的同步更新。
- **统计理论基础**：基于卡尔曼滤波的统计推断理论，提供可解释的图结构演化机制。

### 计算效率

相比 Transformer 的 $O(n^2)$ 复杂度，STG-Mamba 实现 **线性 $O(n)$ 复杂度**，在大型图网络上显著降低 FLOPs 和推理时间。[^src-stg-mamba]

## 实验结果

在三个公开 STG 基准数据集上进行评估：[^src-stg-mamba]

- **PeMS04**（加州高速路网，307 个传感器，交通速度预测）
- **HZMetro**（杭州地铁进出站流量，80 个站点）
- **KnowAir**（中国空气质量监测站）

STG-Mamba 在 RMSE/MAE/MAPE 三项指标上 **一致超越所有基线方法**（含 STGCN、STSGCN、STG-NCDE、DDGCRN、ASTGNN、[[pdformer|PDFormer]]、STAEformer、MultiSPANS）。除 PeMS04 的 MAPE 指标上 STAEformer 略优外，其余所有指标均达到 SOTA。可视化分析表明 STG-Mamba 的预测曲线最贴近真实值波形，峰值捕捉更精准。

## 相关链接

- [[source-diffstg|DiffSTG]] — 最早的 STG 概率扩散预测模型（AAAI 2023），STG-Mamba 的目标是在 Mamba 框架下实现类似或更好的预测质量
- [[source-s-mamba|S-Mamba]] — 首个 Mamba 时序预测基线，双向 Mamba 架构（Neurocomputing 2024）
- [[source-dst-mamba|DST-Mamba]] — 分解式时空 Mamba 长时交通预测（AAAI 2025）
- [[mamba|Mamba]] — 选择性状态空间模型

[^src-stg-mamba]: [[source-stg-mamba]]
