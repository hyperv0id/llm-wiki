---
title: "EM-Diffusion Interleaving (EM 交替式扩散插补)"
type: technique
tags:
  - data-imputation
  - diffusion-models
  - expectation-maximization
  - tabular-data
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# EM-Diffusion Interleaving（EM 交替式扩散插补）

**EM-Diffusion Interleaving** 指 [[diffputer|DiffPuter]]（ICLR 2025）提出的训练机制：把缺失值 $x^{mis}$ 当作隐变量，在「训练扩散模型拟合完整数据联合密度」与「用该模型条件采样更新缺失值」之间交替迭代，使 EM 算法的两个步骤分别落到扩散模型的训练与采样上（DiffPuter 第 4 节）[^src-diffputer]。

## 问题 → 机制

生成式插补需要估计缺失与观测数据的联合分布，但缺失部分未知，训练数据的不完整性使密度估计有内在误差（DiffPuter 第 1 节称之为 incomplete likelihood nature）；同时扩散模型只支持无条件采样、缺少条件推断接口（第 1 节）[^src-diffputer]。EM 提供了一条路径（DiffPuter 第 3.2 节）[^src-diffputer]：

- **M 步**：固定 $x^{mis}$，$\theta^* = \arg\max_\theta p_\theta(x^{obs}, x^{mis})$。DiffPuter 用 VE-SDE 简化版扩散实现：score matching 损失是数据负对数似然的上界（Remark 2，引 Song et al. 2021a 的推论），因此训练后的 $p_\theta$ 近似最大似然估计（第 4.1 节）[^src-diffputer]。
- **E 步**：固定 $\theta$，$x^{mis*} = \mathbb{E}_{x^{mis}\sim p(x^{mis}\mid x^{obs},\theta)}[x^{mis}]$。实现为 RePaint 式混合采样：观测维取 $x^{obs}$ 的前向加噪、缺失维取反向去噪，按掩码合并（式 5-7）；Theorem 1 证明 $\Delta t\to 0$ 时样本精确来自 $p_\theta(x\mid x^{obs})$，$N$ 次采样取均值即 EAP 估计（第 4.2 节）[^src-diffputer]。

## 与相邻方法的区分

- **MCFlow**（归一化流 + 迭代）：以最大似然而非期望恢复缺失值，条件插补靠软正则而非精确条件采样（DiffPuter 第 2 节的区分）[^src-diffputer]。
- **TabCSDI / MissDiff**（一步式扩散插补）：单次训练 + 单次插补，不处理训练集缺失导致的密度估计误差（第 2 节）[^src-diffputer]。消融显示 $k=1$（即纯扩散、无 EM 迭代）只达次优，4-5 次迭代才收敛（图 3）[^src-diffputer]。
- **[[prdim|PRDIM]]**（arXiv 2026）：沿用 EM 框架但改为 hard EM，并在 E 步加入模式识别器引导以显式建模缺失掩码分布、处理 MNAR；PRDIM 论文将 DiffPuter 的做法表述为 soft EM 并作为基线对比[^src-prdim]。

## 适用范围

该机制要求扩散过程保持特征维度与位置（混合采样按维度对齐），适用于表格等维度固定的表示；DiffPuter 的评测全部在表格数据上进行（第 4.2、5.1 节）[^src-diffputer]。

[^src-diffputer]: [[source-diffputer]]
[^src-prdim]: [[source-prdim]]
