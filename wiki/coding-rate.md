---
title: "Coding Rate"
type: concept
tags:
  - information-theory
  - compression
  - representation-learning
  - subspace-learning
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: high
status: active
---

# Coding Rate

**Coding Rate**（编码率）是信息论中衡量描述一个数据分布所需比特数的度量，在 MCR² 目标和 CBSA 中用于量化 token 的"紧凑程度"[^src-cbsa]。

## 定义

对于中心化的数据矩阵 Z ∈ ℝ^(d×N)（d 维特征，N 个样本），在量化精度 ε > 0 下的编码率定义为：

R(Z) = d/2 · log det(I_N + (1/(Nε²)) ZᵀZ)

其中 (1/N)ZᵀZ 是样本的协方差矩阵[^src-cbsa]。

## 直观理解

### 1. 几何视角

编码率衡量在 d 维空间中，用半径为 ε 的球覆盖 N 个数据点所需的"体积"（对数尺度）：

- 数据分布越**紧凑**（方差小）→ 所需覆盖体积小 → 编码率**低**
- 数据分布越**分散**（方差大）→ 所需覆盖体积大 → 编码率**高**[^src-cbsa]

### 2. 信息论视角

编码率可以理解为：使用最优量化方案描述这 N 个点所需的平均比特数的 d 倍（常数��子源于 d 维空间）[^src-cbsa]。

### 3. 奇异值视角

通过矩阵分解，编码率可写为奇异值的函数：

R(Z) = (d/2) · Σ log(1 + σi²/(Nε²))

其中 σi 是 Z 的奇异值。当 σi² ≫ Nε² 时，log(1+x) ≈ log x；当 σi² ≪ Nε² 时，贡献趋于 0[^src-cbsa]。

## 在 MCR² 中的作用

MCR² 目标使用两个编码率项：

R(Z) - Σ_k R(U_kᵀZ)

### 展开项：R(Z)

- 在 ambient 空间 ℝ^d 中编码所有 token
- 防止所有 token 坍缩到原点（否则 R(Z) → -∞）
- 鼓励保持信息[^src-cbsa]

### 压缩项：R(U_kᵀZ)

- 将 token 投影到子空间 U_k 后编码
- 投影丢弃了与子空间正交的信息
- 鼓励 token 在子空间内紧凑排列[^src-cbsa]

**目标**：最小化压缩项（紧凑），同时保持展开项（防止坍缩）[^src-cbsa]。

## 在 CBSA 中的应用

### 约束放松

CBSA 引入代表性 token Q = q(Z)，用不等式约束放松精确等式：

|R(U_kᵀQ) - R(U_kᵀZ)| ≤ τ

这意味着：
- 压缩代表 Q ⇔ 压缩所有 token（在容忍度 τ 内）
- 只需计算 m 个代表的编码率 vs N 个 token[^src-cbsa]

### 梯度步骤

CBSA 对压缩项 R_c(Q|U^[K]) = Σ_k R(U_kᵀQ) 执行梯度下降步，这个梯度步骤直接对应 CBSA 的收缩操作[^src-cbsa]。

## 归一化编码率

论文还引入**归一化编码率**：

R̃(Z) = R(Z̄) / log N,  Z̄ = Z / ||Z||_F

性质：
- 对输入的缩放不变（归一化后测量角度分布，而非幅度）
- 取决于 token 之间的**角度**分布，而非范数
- 用于分析不同层之间的压缩效果对比[^src-cbsa]

## 与其他度量对比

| 度量 | 定义 | 用途 |
|------|------|------|
| **编码率** | (d/2)log det(I + (1/(Nε²))ZZᵀ) | MCR² 目标的核心 |
| 方差 | tr(ZZᵀ)/N | 线性度量，不够敏感 |
| 熵 | -Σ p log p | 需要离散化 |
| 行列式 | det(ZZᵀ) | 非归一化，受数据量影响 |

编码率的优势：同时考虑所有奇异值，在 d 维空间中自然归一化[^src-cbsa]。

## 引用

[^src-cbsa]: [[source-cbsa]]