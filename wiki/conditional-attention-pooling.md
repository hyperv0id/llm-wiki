---
title: "Conditional Attention Pooling (CAP)"
type: technique
tags:
  - time-series
  - attention
  - pooling
  - covariate-fusion
  - TFT
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Conditional Attention Pooling (CAP)

## 概述

**条件注意力池化 (Conditional Attention Pooling, CAP)** 是 UniCA 框架中用于融合协变量信息的核心机制，灵感来自 TFT (Temporal Fusion Transformer)[^src-unca]。

CAP 的核心思想是：根据静态协变量的上下文，动态地从动态协变量中选择最相关的信息，然后将其池化为固定维度的表示。

---

## 背景：从 TFT 到 UniCA

### TFT 中的 Variable Selection

TFT (Temporal Fusion Transformer) 提出了变量选择机制，用于：
1. 选择最相关的输入特征
2. 学习跨实体的一致性表示

### UniCA 中的适配

UniCA 简化并改进了 TFT 的设计：
- **保留核心**：条件注意力机制
- **简化实现**：移除复杂的门控网络
- **适配 TSFM**：针对时间序列基础模型的架构调整

---

## 数学定义

### 输入

- **动态协变量嵌入**：$E_{C_{1:T}} \in \mathbb{R}^{P \times M \times d}$
  - $P$：时间 token 数量
  - $M$：协变量数量
  - $d$：嵌入维度

- **静态协变量嵌入**：$E_S \in \mathbb{R}^{N \times d}$
  - $N$：静态特征数量（如果没有则为 0，向量置零）

### 核心运算

$$Z_{C_{1:T}} = \text{CAP}(E_{C_{1:T}} | E_S) := \text{softmax}(A)V$$

其中：

$$A = \text{GRN}(\text{flat}(E_{C_{1:T}}), E_S)$$
$$V = \text{GRN}(E_{C_{1:T}})$$

### 维度解释

| 符号 | 形状 | 含义 |
|------|------|------|
| $E_{C_{1:T}}$ | $(P, M, d)$ | 动态协变量的 token 嵌入 |
| $\text{flat}(E_{C_{1:T}})$ | $(P, M \cdot d)$ | 展平后的协变量特征 |
| $A$ | $(P, 1, M)$ | 每个 token 对各协变量的注意力分数 |
| $V$ | $(P, M, d)$ | 协变量的值表示 |
| $Z_{C_{1:T}}$ | $(P, d)$ | 池化后的协变量表示 |

---

## 组件详解

### GRN (Gated Residual Network)

GRN 是带残差连接的门控 MLP，是 TFT 的核心组件：

```
Input (dim: d_in)
    ↓
GLU (Gate Linear Unit)
    ↓
Dense (d_in → d_hidden)
    ↓
ReLU / GELU
    ↓
Dense (d_hidden → d_out)
    ↓
Add (残差连接)
    ↓
LayerNorm
    ↓
Output (dim: d_out)
```

**GRN 的优势**：

1. **门控机制**：学习何时使用输入特征
2. **残差连接**：避免梯度消失
3. **层归一化**：训练稳定性
4. **可学习非线性**：比简单线性变换更强的表达能力

### 展平操作 (flat)

$$\text{flat}(E_{C_{1:T}}) \in \mathbb{R}^{P \times (M \cdot d)}$$

将最后两个维度展平，为注意力计算做准备。

### Softmax 注意力

$$\text{softmax}(A) \in \mathbb{R}^{P \times 1 \times M}$$

沿协变量维度计算 softmax，得到归一化的注意力权重。

---

## 融合公式

CAP 的输出通过 GLU (Gated Linear Unit) 与目标序列融合：

$$\tilde{Z} = Z + \text{GLU}(Z_{C_{1:T}})$$

### GLU 机制

GLU (Gated Linear Unit) 定义为：

$$\text{GLU}(X) = \sigma(W_1 X + b_1) \odot (W_2 X + b_2)$$

其中：
- $\sigma$：sigmoid 门控函数
- $W_1, W_2$：投影矩阵
- $\odot$：逐元素乘法

**作用**：学习权衡协变量信息的影响程度。

---

## 应用场景

### 场景 1：Pre-Fusion 模块

**位置**：Tokenizer → Temporal Encoder

**输入**：
- $Y_{1:T}$：历史目标序列
- $C_{1:T}$：历史协变量
- $S$：静态协变量

**处理流程**：
```
1. Tokenize Y: Z = T(Y)
2. Tokenize C: E_C = T(C)
3. Encode S: E_S = ρ(S)
4. CAP 池化: Z_C = CAP(E_C | E_S)
5. GLU 融合: Z_tilde = Z + GLU(Z_C)
6. 编码: H_tilde = E(Z_tilde)
```

**目的**：在时间编码前，将历史协变量信息融入目标表示，使编码器能够捕获联合动态。

### 场景 2：Post-Fusion 模块

**位置**：Temporal Encoder → Predictor

**输入**：
- $H$：编码后的目标表示
- $C_{T+1:T+H}$：未来已知协变量
- $S$：静态协变量

**处理流程**：
```
1. Tokenize C_future: E_C_fut = T(C_future)
2. CAP 池化: Z_C_fut = CAP(E_C_fut | E_S)
3. 自注意力融合: [H_hat, Z_C_fut] = SelfAttn([H, Z_C_fut])
4. 预测: Y_hat = P(H_hat)
```

**目的**：在时间编码后，将未来协变量信息融入预测表示。

---

## 可解释性

### 注意力权重的含义

CAP 的注意力权重揭示了模型如何动态选择协变量：

1. **时间维度变化**：注意力权重在时间步上动态变化，某些协变量在不同时刻的重要性不同

2. **协变量选择**：某些协变量可能始终获得较高权重（如与目标强相关的变量），而其他协变量仅在特定时刻重要

3. **信息互补**：模型学习协变量信息是对目标信息的**补充**而非重复

### 可视化分析

论文中的注意力可视化（图 5c）显示：
- 特定协变量（如 Covariate 13）持续获得最高权重
- 这些协变量具有丰富的时间模式，与目标强相关
- 目标本身的注意力不过高，说明融合模块学会利用协变量补充信息[^src-unca]

---

## 与其他方法的对比

### vs. 直接拼接

| 方法 | 处理方式 | 缺点 |
|------|---------|------|
| 直接拼接 | 将协变量与目标在维度上拼接 | 维度爆炸，协变量权重固定 |
| **CAP** | 注意力加权池化 | 动态权重，自动选择 |

### vs. 简单平均

| 方法 | 处理方式 | 缺点 |
|------|---------|------|
| 简单平均 | 所有协变量等权重 | 无法区分重要性 |
| **CAP** | 注意力加权 | 学习加权 |

### vs. TFT Variable Selection

| 特性 | TFT VS | CAP |
|------|--------|-----|
| 复杂度 | 复杂多层 | 简化单层 |
| 实体一致性 | ✓ | ✗ |
| 时间动态 | ✓ | ✓ |
| 静态条件 | ✓ | ✓ |

---

## 相关技术

- [[unified-covariate-adaptation]] — UniCA 框架
- [[gated-linear-units]] — GLU 机制
- [[timesnet]] — 时间序列基础模型
- [[covariate-homogenization]] — 协变量同质化

---

## 引用

[^src-unca]: [[source-unca]]