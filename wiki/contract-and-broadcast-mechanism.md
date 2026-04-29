---
title: "Contract-and-Broadcast Mechanism"
type: technique
tags:
  - attention
  - cbsa
  - algorithm-unrolling
  - compression
  - token-mixer
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Contract-and-Broadcast Mechanism

**Contract-and-Broadcast**（收缩-广播）是 CBSA 的核心操作机制，通过两步实现 token 压缩：先收缩代表性 token，再将收缩结果广播回所有输入 token[^src-cbsa]。

## 机制概述

### 两阶段设计

```
Input Tokens Z
    ↓
[1] Representative Extraction (代表性 token 提取)
    ↓
Representative Tokens Q + Attention Matrix A
    ↓
[2] Representative Contraction (代表性 token收缩)
    ↓
Contraction Direction
    ↓
[3] Contraction Broadcast (收缩广播)
    ↓
Output Tokens Z'
```

### 直观理解

1. **提取**：从输入 token 中提取少数代表性 token Q（m << N）
2. **收缩**：对代表性 token 执行自注意力得到收缩方向
3. **广播**：将收缩方向通过注意力矩阵广播回所有输入 token

这实现了"压缩所有 token 通过收缩少数代表"的核心思想[^src-cbsa]。

## 数学细节

### 代表性 token 提取

使用 cross-attention 近似编码率约束来提取代表性 token：

$$U_k^\top Q = U_k^\top Q_{ini} + \eta \cdot U_k^\top Z \cdot \text{softmax}(U_k^\top Z \cdot U_k^\top Q_{ini})$$

其中：
- $Q_{ini} = \text{sg}(\text{avgpool}(U_k^\top Z))$：初始猜测（平均池化）
- $\eta$：可学习步长
- $\text{sg}(\cdot)$：停梯度算子确保初始化不影响后续推导
- 该操作对应系数矩阵 $A_k$ 的计算[^src-cbsa]

### 代表性 token 收缩

对代表性 token 执行梯度下降步骤（压缩项的梯度）：

$$\text{Contraction} = U_k^\top Q \cdot (I_m + \frac{p}{m\epsilon^2}(U_k^\top Q)^\top (U_k^\top Q))^{-1}$$

为避免昂贵矩阵求逆，用 Gram 矩阵 + softmax 近似：

$$\text{Contraction} \approx U_k^\top Q \cdot \text{softmax}((U_k^\top Q)^\top (U_k^\top Q))$$

这实际上是一个 **self-attention 操作**，query/key/value 都是 $U_k^\top Q$[^src-cbsa]。

### 收缩广播

利用提取阶段的注意力矩阵 $A_k$ 将收缩广播回所有输入 token：

$$\text{Broadcast} = A_k^\top \cdot \text{Contraction}$$

$$\text{CBSA}(Z) = U_k \cdot \text{Broadcast}$$

**关键洞察**：广播项复用提取阶段的注意力矩阵，避免额外计算[^src-cbsa]。

## 与标准注意力的对比

| 方面 | 标准 Softmax Attention | CBSA Contract-and-Broadcast |
|------|----------------------|---------------------------|
| 复杂度 | O(N²d) | O(Nmd)（m << N） |
| Token 交互 | 所有 N² 对 | 通过 m 个代表间接交互 |
| 信息流向 | 完全连接 | 通过代表性 token 路由 |
| 可解释性 | 黑盒 | 梯度步骤可解释 |

## Contract-and-Broadcast 变体

### 1. Softmax 变体（MSSA）

当 Q = Z, m = N 时，收缩变为标准 softmax attention：

$$\text{CBSA}(Z) = \sum_k U_k^\top \text{softmax}(U_k^\top Z Z^\top U_k) U_k^\top Z$$

这正是 MSSA[^src-cbsa]。

### 2. Linear Attention 变体

当代表性 token 取为主方向（SVD 后的左奇异向量）时，收缩变为线性 attention：

$$\text{Contraction} = U_k^\top Z \cdot \phi((U_k^\top Z)^\top (U_k^\top Z))$$

其中 $\phi(\lambda_i) = \epsilon^2 / (\epsilon^2 + \lambda_i)$ 是谱上的函数[^src-cbsa]。

### 3. Channel Attention 变体

当代表性 token 固定为标准基（$L_k = I_p$）时收缩变为通道缩放：

$$\text{Contraction} = U_k^\top D_k U_k^\top Z$$

其中 $D_k = \text{Diag}[f(u_{ki}^\top Z (u_{ki}^\top Z)^\top)]$ 是通道wise 缩放因子[^src-cbsa]。

### 4. Agent Attention 变体

当移除收缩步骤时等效于 Agent Attention：

$$\text{Agent}(Z) = \text{Broadcast without Contraction}$$

这解释了 Agent Attention 的"不完全 token mixer"特性[^src-cbsa]。

## 优势

1. **线性复杂度**：固定 m 时 O(N) 而非 O(N²)
2. **可解释性**：每步对应优化目标的梯度步骤
3. **统一框架**：涵盖多种注意力变体
4. **可学习步长**：η、κ 等参数从数据学习而非手动调参[^src-cbsa]

## 引用

[^src-cbsa]: [[source-cbsa]]