---
title: "Pyraformer"
type: entity
tags:
  - time-series
  - forecasting
  - transformer
  - efficient-attention
  - multi-scale
  - pyramidal-attention
  - ICLR-2022
  - ant-group
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# Pyraformer

**Pyraformer** 是由蚂蚁集团（Ant Group）、上海交大和 TU Wien 联合提出的高效 Transformer 架构，发表于 **ICLR 2022 Oral**。它首创金字塔注意力机制（Pyramidal Attention），在时间序列中同时捕获短程和长程时间依赖，实现 **O(L) 时间空间复杂度**与 **O(1) 最大信号传播路径长度**，是首个在理论上同时达成这两个理想的 Transformer 变体 [^src-pyraformer]。

## Overview

时间序列预测的核心挑战在于构建一个能在不同时间尺度上捕获依赖关系的高效模型。此前的方法始终存在一个根本性妥协：

- **RNN/CNN**：O(L) 复杂度，但最大信号传播路径为 O(L)，无法高效捕获远程依赖
- **Transformer**：O(1) 最大路径，但 O(L^2) 复杂度，无法处理长序列
- **稀疏 Transformer 变体（LogTrans, Informer, Longformer, Reformer, ETC）**：取折中，但多数无法同时达到 O(1) 路径和 O(L) 复杂度 [^src-pyraformer]

Pyraformer 用**多分辨率金字塔图**打破这一僵局：在粗尺度上描述远程依赖天然更简洁（图意义上的"parsimonious"），将计算量从 O(L^2) 压至 O(L)，同时保持全局感受野 [^src-pyraformer]。

## 核心架构

Pyraformer 由两个关键模块组成：**CSCM** 构建多分辨率树结构，**PAM** 在其上进行高效消息传递。

### CSCM（Coarser-Scale Construction Module）

CSCM 负责初始化金字塔图中的粗尺度节点。输入嵌入序列后，在时间维度上施加 bottleneck 卷积（先降维压缩、kernel size=C, stride=C 逐层卷积、再升维恢复），自底向上生成尺度 s 处长度为 L/C^(s-1) 的序列。不同尺度序列串联形成 C-叉树 [^src-pyraformer]。

CSCM 采用 bottleneck 结构，卷积 kernel size 和 stride 均为 C，参数量仅比无 CSCM 增加约 5%。相比 max pooling 或 average pooling，bottleneck 卷积以 +1.51% MSE 的微小代价换取 **90% 参数量的减少** [^src-pyraformer]。

### PAM（Pyramidal Attention Module）

PAM 是 Pyraformer 的心脏，本质上是在金字塔图上进行消息传递。每个节点 n_l^(s) 的注意力邻居集合由三部分组成 [^src-pyraformer]：

| 邻居类型 | 描述 |
|----------|------|
| 同尺度相邻 A 个节点 | 捕获该分辨率下的短期依赖 |
| C 个子节点（细尺度） | 来自更低层多分辨率树节点的汇总信息 |
| 1 个父节点（粗尺度） | 连接到更高层多分辨率树节点的汇总信息 |

通过堆叠 N 层 PAM，信息在金字塔图中双向流动。Lemma 1 给出最粗尺度节点获得全局感受野的充分条件。Proposition 1-2 证明：固定 A 为小常数时，复杂度为 O(L)，最大路径长度为 O(1) [^src-pyraformer]。

由于 PAM 的稀疏注意力模式不被 PyTorch/TensorFlow 原生支持，作者基于 **TVM** 实现了定制 CUDA kernel，实测计算时间和内存显著降低 [^src-pyraformer]。

### 预测模块

- **单步预测**：在历史序列末追加 end token，PAM 编码后收集各尺度最后节点，拼接经 FC 层预测
- **多步预测方案一**（推荐）：同上但 FC 层直接映射到全部 M 个未来时间点
- **多步预测方案二**：使用两层 full attention decoder，预测 token 作为 query，encoder 输出作为 key/value [^src-pyraformer]

方案一在多数实验中优于方案二，因为 decoder 的 full attention 无法区分多分辨率特征，而单 FC 层可自动利用这些特征 [^src-pyraformer]。

## 性能表现

### 单步预测

在 Electricity, Wind, App Flow 三数据集上，Pyraformer 以 **最少的 Q-K dot product 对** 取得最优 NRMSE 和 ND。具体而言 [^src-pyraformer]：

- Q-K pairs 比 LogTrans 少 **65.4%**
- Q-K pairs 比 full attention 少 **96.6%**
- CSCM 仅带来约 5% 的参数增量
- 在稀疏数据集（Wind）上，稀疏注意力机制一致优于 full attention（防止过拟合）

### 长期多步预测

在 [[lstf|LSTF]] 标准 benchmark（ETTh1, ETTm1, Electricity）上 [^src-pyraformer]：

| 指标 | ETTh1 (168/336/720) |
|------|---------------------|
| MSE vs Informer | **-24.8% / -28.9% / -26.2%** |
| 所有预测长度的 Q-K pairs | 最少 |

在合成数据（多段正弦函数 + 长程高斯过程）上，Pyraformer 以大幅优势领先 SOTA，且根据已知周期设定不同尺度 C 值可继续提升性能 [^src-pyraformer]。

### 计算效率

12GB Titan Xp GPU 实测 [^src-pyraformer]：

| 序列长度 | Full Attention | Informer | Pyraformer-TVM |
|----------|----------------|----------|-----------------|
| 5,800 | OOM | -- | **1 GB** |
| 20,000 | -- | OOM | **1.91 GB, 0.082s/batch** |

内存和计算时间为序列长度的近线性函数，验证了理论 O(L) 复杂度。

## 消融实验关键发现

1. **A 应固定为小常数（3 或 5）**，C 随 L 增大；一旦顶层节点获得全局感受野，继续增大 A 不再带来增益
2. **PAM 对准确预测至关重要**——仅保留 CSCM 时性能大幅下降
3. **更长历史提升精度**，但增益在历史提供足够周期信息后趋于饱和
4. Bottleneck 卷积 CSCM 是最优的尺度构建方式 [^src-pyraformer]

## 超参数选择指南

Pyraformer 的超参数包括 S（尺度数）、N（注意力层数）、A（同尺度相邻节点数）、C（子节点数）。推荐策略：先根据计算资源确定 N，再根据时间序列粒度确定 S（如小时观测设 S=4，对应日/周/月周期），A 取 3 或 5，最后用验证集从满足全局感受野条件的候选值中选 C [^src-pyraformer]。

## 历史地位与影响

Pyraformer 是 [[lstf|LSTF]] 演化链中**效率与结构并重**的关键节点：

- **首创金字塔注意力**——在时间序列 Transformer 中引入多分辨率建模，将 O(L) 复杂度和 O(1) 路径长度首次兼得
- **先于 [[autoformer|Autoformer]]（NeurIPS 2021）和 [[fedformer|FEDformer]]（ICML 2022）**——但 Pyraformer 走的是多尺度图结构路线，不同于 Autoformer 的分解+Fourier 和 FEDformer 的频率域路线
- **为多分辨率时间序列建模提供理论框架**——Lemma 1、Proposition 1-2 给出了全局感受野和复杂度的明确条件

### 连接

- **[[informer|Informer]]**（AAAI 2021 Best Paper）：Pyraformer 的直接对比基线，在 ETTh1 上 MSE 降低 24.8%-28.9%。Informer 的 ProbSparse attention 为 O(L log L)，Pyraformer 降至 O(L) [^src-pyraformer]
- **[[autoformer|Autoformer]]**（NeurIPS 2021）：同为 ICLR 2021-2022 时期的 Transformer 创新，但走分解 + Auto-Correlation 路线，关注季节性趋势分离
- **[[fedformer|FEDformer]]**（ICML 2022）：另一个达到 O(L) 的模型，但通过频率域注意力实现，Pyraformer 通过多尺度图结构实现
- **[[patchtst|PatchTST]]**（ICLR 2023）：用 patch tokenization 将序列长度压缩，Pyraformer 的 C-叉树是另一种压缩思路
- **[[itransformer|iTransformer]]**（ICLR 2024）：颠覆性翻转 Transformer 维度，在变量维度做 attention——正交于 Pyraformer 的时间多分辨率思路

### 批判视角

尽管 Pyraformer 在效率上取得突破，但存在局限性：
- **依赖 TVM 定制 CUDA kernel**——增加了部署复杂度，不如后续 O(L) 模型（FEDformer, PatchTST）易用
- **CSCM 引入额外参数**（约 +5%），且 C 值的选择依赖人工和验证集
- **[[ltsf-linear|LTSF-Linear]]**（Zeng et al., 2022）的批判间接适用于 Pyraformer——简单线性模型在多个 benchmark 上超越 Transformer，Pyraformer 的多尺度设计是否能真正优于线性基线值得更多审视

## 局限性

- 仅考虑 A 和 S 固定、C 随 L 增长的超参数配置，未探索自适应学习
- TVM 定制 kernel 的依赖限制了即插即用的便利性
- 未来方向包括：自适应学习超参数，扩展到 NLP/CV 领域 [^src-pyraformer]

[^src-pyraformer]: [[source-pyraformer]]
