---
title: "Unified Covariate Adaptation (UniCA)"
type: concept
tags:
  - time-series
  - foundation-model
  - adaptation
  - covariate
  - iclr2026
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Unified Covariate Adaptation (UniCA)

## 概述

**Unified Covariate Adaptation (UniCA)** 是 ICLR 2026 论文提出的统一协变量适应框架，旨在将时间序列基础模型 (TSFMs) 扩展到处理一般协变量感知预测任务[^src-unca]。

## 背景问题

### 时间序列基础模型 (TSFMs)

TSFMs（如 Chronos、Moirai、TimesFM、MOMENT）通过大规模预训练学习可迁移的时间表示，在零样本场景下表现出色。然而，它们存在一个关键限制：

> **预训练设计针对单变量实数值序列，无法直接处理外生协变量**[^src-unca]

### 协变量类型

| 类型 | 定义 | 示例 |
|------|------|------|
| **同质协变量** | 与目标序列形式相同的实值时间序列 | 其他相关的时间序列 |
| **异构协变量** | 与目标序列形式不同的变量 | 分类变量（商品ID）、图像、文本 |

### 现有方法的局限性

1. **专用模型**：DeepAR、TFT、TiDE 等需要从零训练，缺乏预训练泛化能力
2. **现有 TSFMs 适配**：
   - TimesFM：仅使用辅助回归器进行残差修正
   - ChronosX：通过线性变换注入协变量，但仅适用于点式 TSFM
   - Moirai：将序列和协变量展平为联合序列，使用 variate ID 区分

**以上方法都无法处理异构协变量**，这正是 UniCA 要解决的核心问题。

---

## 核心思想

UniCA 遵循两个关键设计原则：

### 原则 1：协变量同质化 (Covariate Homogenization)

将所有类型的协变量转换为统一的高阶同质时间序列表示，从而**弥合协变量与目标时间序列之间的异构性差距**。

### 原则 2：模块化融合 (Modular Fusion)

将 TSFM 架构分解为可解释的阶段，并在适当位置插入基于注意力的融合模块，**避免干扰预训练的模型动态**。

---

## 架构详解

UniCA 将标准 TSFM 分解为三个可解释的组件：

```
输入 → Tokenizer → Temporal Encoder → Predictor → 输出
              ↑              ↑              ↑
         Pre-Fusion    (保持不变)    Post-Fusion
```

### 1. Tokenizer (分词器)

$$Z = \mathcal{T}(Y_{1:T})$$

将原始时间序列输入 $Y \in \mathbb{R}^{T \times 1}$ 转换为 token 序列 $Z \in \mathbb{R}^{P \times d}$，其中：
- $P$：沿时间维度的 token 数量
- $d$：token 维度

分词方式包括：
- **基于 Patch**：如 TimesFM、Chronos（重叠 patch）
- **基于 Point**：如 Chronos-T5（逐点）

### 2. Temporal Encoder (时间编码器)

$$H = \mathcal{E}(Z)$$

编码器处理 token 序列以提取高阶时间模式。最常见的是 Transformer 架构。

### 3. Predictor (预测器)

$$\hat{Y}_{T+1:T+H} = \mathcal{P}(H)$$

预测器利用编码表示生成未来预测。对于解码器-only 架构（如 LLM），线性输出层即为预测器。

---

## 核心组件

### 组件 1：协变量同质化模块 (Covariate Homogenizer, CH)

**目的**：将异构协变量转换为统一的同质时间序列表示

#### 分类协变量处理

1. 使用 embedding 层将离散类别映射为连续向量
2. 每个类别 ID 通过查找表获得 d 维嵌入表示

#### 多模态协变量处理

**图像协变量**：
- 使用简单的 4 层 CNN（针对 64×64×4 卫星图像优化）
- 提取空间特征后投影为时间序列表示

**文本协变量**：
- 使用 GIST 编码器提取文本特征
- 通过线性层投影为同质表示

#### 数学公式

给定异构协变量的特征表示 $H^{(het)}_{1:T+H}$，同质化过程为：

$$C^{(het)}_{1:T+H} = CH(H^{(het)}_{1:T+H})$$

其中：
- $CH$ 是协变量同质化器（默认使用简单线性层）
- $C^{(het)}_{1:T+H} \in \mathbb{R}^{(T+H) \times d_{het}}$
- $d_{het}$ 是可调超参数，默认值为 4[^src-unca]

### 组件 2：Pre-Fusion 模块

**位置**：Tokenizer 和 Temporal Encoder 之间

**目的**：在编码前将历史协变量信息与目标序列融合，使编码器能够捕获时间序列与历史外部因素的联合动态。

#### 工作流程

1. **协变量编码**：
   $$E_{C_{1:T}} = \mathcal{T}(C_{1:T}), \quad E_S = \rho(S)$$

2. **条件注意力池化 (CAP)**：
   $$Z_{C_{1:T}} = \text{CAP}(E_{C_{1:T}} | E_S) := \text{softmax}(A)V$$
   
   其中：
   - $A = \text{GRN}(\text{flat}(E_{C_{1:T}}), E_S)$
   - $V = \text{GRN}(E_{C_{1:T}})$

3. **门控融合**：
   $$\tilde{Z} = Z + \text{GLU}(Z_{C_{1:T}})$$

4. **编码**：
   $$\tilde{H} = \mathcal{E}(\tilde{Z})$$

### 组件 3：Post-Fusion 模块

**位置**：Temporal Encoder 和 Predictor 之间

**目的**：将未来已知协变量信息融入编码表示，因为未来协变量能直接提供未来条件的洞察。

#### 工作流程

1. **未来协变量编码**：
   $$E_{C_{T+1:T+H}} = \mathcal{T}(C_{T+1:T+H})$$

2. **条件注意力池化**：
   $$Z_{C_{T+1:T+H}} = \text{CAP}(E_{C_{T+1:T+H}} | E_S)$$

3. **自注意力融合**：
   $$[\hat{H}, \hat{Z}_{C_{T+1:T+H}}] = \text{SelfAttn}([\tilde{H}, Z_{C_{T+1:T+H}}])$$

4. **预测**：
   $$\hat{Y}_{T+1:T+H} = \mathcal{P}(\hat{H})$$

---

## 训练细节

### 损失函数

UniCA 与各种 TSFMs 兼容，使用与基础模型预训练相同的损失函数：

| TSFM | 损失函数 |
|------|----------|
| Chronos, TimesFM | Quantile Loss |
| Time-MoE | Huber Loss |
| Moirai | Negative Log Likelihood (NLL) |

### 归一化

为训练稳定性，对每个目标实例按其历史均值和标准差进行归一化（实例归一化，参考 RevIN）[^src-unca]。

### 学习率

从 $\{10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}\}$ 中选择基于验证性能。

---

## 性能表现

### 单模态协变量数据集 (12 个数据集平均)

| 方法 | MAPE | 相对提升 |
|------|------|----------|
| TimesFM (Zero-shot) | 0.598 | - |
| TimesFM (SFT) | 0.600 | -0.3% |
| TimesFM (UniCA) | 0.522 | -12.7% |
| Chronos-Bolt (Zero-shot) | 0.598 | - |
| Chronos-Bolt (UniCA) | **0.506** | **-15.4%** |
| Time-MoE (UniCA) | 0.514 | -14.0% |

### 多模态数据集

**MMSP (图像-时序)**：
| 方法 | MAE | 提升 |
|------|-----|------|
| TimesFM (Zero-shot) | 0.778 | - |
| TimesFM (UniCA) | **0.634** | **-6.5%** |
| Chronos-Bolt (UniCA) | 0.648 | -5.9% |

**Time-MMD (文本-时序)**：
| 方法 | MAPE | 提升 |
|------|------|------|
| TimesFM (Zero-shot) | 0.900 | - |
| TimesFM (UniCA) | **0.656** | **-5.9%** |
| Chronos-Bolt (UniCA) | 0.601 | -13.0% |

---

## 效率分析

### 推理时间

UniCA 引入的计算开销极小：
- 推理时间增加 <10%（与冻结基线相比）

### 可训练参数量

| 模型 | 原始参数 | +UniCA 额外参数 |
|------|----------|----------------|
| Chronos-Bolt | ~800M | ~2M (0.25%) |
| TimesFM | ~200M | ~1M (0.5%) |

---

## 兼容性

UniCA 适用于大多数主流 TSFMs：

| TSFM | 架构类型 | 支持状态 |
|------|----------|----------|
| Chronos / Chronos-Bolt | Patch-based Transformer | ✓ |
| TimesFM | Patch-based Decoder | ✓ |
| Time-MoE | Mixture of Experts | ✓ |
| Moirai | Patch-based T5 | ✓ |
| MOMENT | Patch-based Transformer | ✓ |

---

## 相关技术

- [[timesnet]] — 时间序列基础模型
- [[conditional-attention-pooling]] — CAP 机制
- [[heterogeneous-covariates]] — 异构协变��
- [[covariate-homogenization]] — 协变量同质化
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测

---

## 引用

[^src-unca]: [[source-unca]]