---
title: "Muon优化器赏析：从向量到矩阵的本质跨越"
type: source-summary
tags:
  - optimizer
  - matrix-optimization
  - spectral-norm
  - theory
created: 2026-04-30
last_updated: 2026-04-30
source_count: 3
confidence: high
status: active
---

# Source Summary: Muon优化器赏析

## 核心论点

本文从理论层面深入分析了 Muon 优化器的数学本质，提出 Muon 体现了向量与矩阵优化的本质差异[^src-kellerjordan-muon-blog][^src-kexue-muon-analysis]。

## 主要理论贡献

### 1. 矩阵符号函数 (msign)

$\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$，其中 $\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ 是 SVD 分解[^src-kexue-muon-analysis]

### 2. 最优正交近似

$$
\text{msign}(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}} \Vert \boldsymbol{M} - \boldsymbol{O} \Vert_F^2
$$

### 3. 谱范数视角

Muon 相当于谱范数（2-范数）约束下的梯度下降[^src-kexue-muon-analysis]：

- 谱范数：$\Vert \boldsymbol{\Phi}\Vert_2 = \max_{\Vert \boldsymbol{x}\Vert_2 = 1} \Vert \boldsymbol{\Phi}\boldsymbol{x}\Vert_2$
- 相比 Frobenius 范数更紧凑，能更好地度量矩阵间的本质差异

### 4. 与其他优化器的关系

| 优化器 | 关系 |
|--------|------|
| SignSGD/Tiger | 当 $\boldsymbol{M}$ 是对角阵时，Muon 退化为 SignSGD[^src-kexue-muon-analysis] |
| Shampoo | $\beta=0$ 时等价[^src-kexue-muon-analysis] |
| Adam | Element-wise 更新 vs 矩阵整体更新 |

### 5. 历史渊源

- 2015 年论文 "Stochastic Spectral Descent for Restricted Boltzmann Machines" 已提出类似算法[^src-kexue-muon-analysis]
- Shampoo (2018) 缓存梯度外积，$\beta=0$ 时与 Muon 等价[^src-kexue-muon-analysis]

## 关键洞见

> 向量与矩阵的本质差异：矩阵有"迹"等特有概念，对角线元素与非对角线元素地位不完全对等。Muon 正是捕捉了这种不对称性。

## 实践考虑

- 计算开销：比 Adam 增加约 5%（作者声称 2%）
- 显存成本：比 Adam 少一组缓存变量
- 需要将 QKV 矩阵分开处理以获得最佳效果

## 参考��献

[^src-kellerjordan-muon-blog]: [[source-kellerjordan-muon-blog]]
[^src-muon-optimizer]: [[source-muon-optimizer]]
[^src-kexue-muon-analysis]: [[source-kexue-muon-analysis]]