---
title: "Representative Token Extraction"
type: technique
tags:
  - attention
  - cbsa
  - token-representation
  - compression
  - landmark
created: 2026-04-29
last_updated: 2026-10-24
source_count: 2
confidence: medium
status: active
---

# Representative Token Extraction

**Representative Token Extraction**（代表性 token 提取）是 CBSA 的关键组件，通过可微映射从输入 token 提取少数代表性 token（representatives），使得压缩代表性 token 等价于压缩所有输入 token[^src-cbsa]。

## 核心思想

### 问题

MCR² 目标的压缩项 $R(U_k^{\top}Z)$ 需要计算所有 $N$ 个 token 的编码率：
- 时间复杂度：$O(N^2)$ 来自 Gram 矩阵
- 空间复杂度：$O(N^2)$ 存储注意力矩阵

### 解决方案

用 $m \ll N$ 个代表性 token $Q$ 替代所有 $N$ 个 token：

$$Q = q(Z), \quad q: \mathbb{R}^{d \times N} \to \mathbb{R}^{d \times m}$$

要求：
1. Q 捕捉 Z 的"信息-几何本质"
2. 压缩 Q 等价于压缩 Z（在容忍度 τ 内）

## 数学框架

### 约束条件

为确保压缩 Q 等价于压缩 Z，施加不等式约束：

$$|R(U_k^\top Q) - R(U_k^\top Z)| \leq \tau, \quad \forall k \in [K]$$

其中 τ 是容忍度参数[^src-cbsa]。

### 代表性 token 的结构

论文采用线性组合形式：

$$U_k^\top Q = U_k^\top Z A_k$$

其中 $A_k \in \mathbb{R}^{N \times m}$ 是系数矩阵。

这意味着：
- 每个代表性 token 是 N 个输入 token 的线性组合
- 代表捕捉了输入 token 在子空间 k 上的投影信息[^src-cbsa]

## 提取方法

### Cross-Attention 近似

论文使用 cross-attention 来提取满足约束的代表性 token：

$$U_k^\top Q = U_k^\top Q_{ini} + \eta \cdot U_k^\top Z \cdot \text{softmax}(U_k^\top Z \cdot (U_k^\top Q_{ini})^\top)$$

**直觉**：
- 这与 cross-attention 形式相同（Q 作为 query，Z 作为 key/value）
- Cross-attention 可以解释为编码率的近似
- 初始化 Qini 通过平均池化得到（简单但有效的全局信息聚合）[^src-cbsa]

### 初始猜测

$$Q_{ini} = \text{sg}(\text{avgpool}(U_k^\top Z))$$

- 对子空间投影后的 token 执行平均池化
- sg(·) 是停梯度算子，确保初始化不影响后续梯度计算
- 提供代表性 token 的初始估计（全局统计信息）[^src-cbsa]

### 可学习步长

$$\eta = \text{learnable parameter}$$

- 步长 η 作为可学习参数，从数据自适应
- 而非手动调参[^src-cbsa]

## 代表性 token 的不同选择 → 不同注意力机制

### 1. 自表达代表（Softmax Attention / MSSA）

$$Q = Z, \quad m = N$$

- 每个输入 token 都是"代表"
- 压缩所有 token = 压缩每个 token 自身
- 对应 MSSA（Multi-head Subspace Self-Attention）[^src-cbsa]

### 2. 正交代表（Linear Attention）

通过对 $U_k^\top Z$ 执行 SVD 得到主方向作为代表：

$$U_k^\top Z = L_k \Sigma_k R_k$$

取 $U_k^\top Q = L_k \Sigma_k$（主方向 + 奇异值）

- 代表捕捉主奇异方向
- 动态适应输入数据[^src-cbsa]

### 3. 固定正交代表（Channel Attention / TSSA）

代表固定为标准基 $L_k = I_p$：

- 不依赖输入数据
- 等效于通道wise 缩放（TSSA）[^src-cbsa]

### 4. 无代表（Agent Attention）

移除代表性 token 提取步骤：
- 等效于仅使用初始化（平均池化）进行信息聚合
- 对应 Agent Attention 的设计[^src-cbsa]

## 与其他"代表"概念对比

| 方法 | 代表选择 | 用途 | 与 CBSA 的关系 |
|------|----------|------|----------------|
| **CBSA 代表** | 通过 cross-attention 从输入提取 | 压缩目标 | 核心组件 |
| **全局 token** | 额外添加的 cls token | 全局信息 | CBSA 的 Q_ini 包含平均池化 |
| **Landmarks** | 随机/学习选择的 token 子集 | 稀疏注意力 | 相似但 CBSA 的代表可学习 |
| **Agent tokens** | 额外添加的可学习 token | 线性 attention | CBSA 的无收缩变体 |

上表的 Landmarks 一行指随机或学习选择的 token 子集。MiTA（arXiv:2602.01219）另给一种取法：landmark query 由该层 $Q$ 做均匀分段的平均池化得到（$m\ll N$ 个），landmark value 由这些 query 对全 KV 做一次 cross-attention 得到，二者构成对全体 query 常开的共享专家；同一批 landmark 再各取 top-$k$ KV 立成专家组，query 侧只做一次 argmax 路由[^src-2602-01219-mita]。在 MiTA 的五维分类里，这种构造与 Agent attention 同记为 landmark testing，差别在路由拓扑。见 [[mita-attention]]、[[fast-weight-scaling-taxonomy]]。

## 优势

1. **显著压缩**：m << N，大幅降低计算量
2. **可微**：通过 cross-attention 实现，可端到端训练
3. **可学习**：初始化方法、系数矩阵均可学习
4. **统一框架**：通过改变代表结构实现不同注意力变体[^src-cbsa]

## 引用

[^src-cbsa]: [[source-cbsa]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]