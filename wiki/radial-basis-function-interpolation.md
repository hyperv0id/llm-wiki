---
title: "Radial Basis Function Interpolation"
type: technique
tags:
  - interpolation
  - spatial-analysis
  - rbf
  - air-quality
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Radial Basis Function (RBF) Interpolation

RBF 插值是一种从离散观测推断连续场的核方法，特别适用于不规则分布的空间数据。[^src-ctenet]

## 原理

$$rbf(x) = \sum_{i=1}^{n_t} \lambda_i^{(t)} \phi(\|x - x_i^{(t)}\|)$$

其中 $\phi(r)$ 是径向基函数，$\lambda_i^{(t)}$ 是通过求解线性系统 $\Phi^{(t)}\lambda^{(t)} = y^{(t)}$ 确定的插值权重，$\Phi_{ij} = \phi(\|x_i - x_j\|)$ 是核矩阵。[^src-ctenet]

## Multiquadric 核

CTENet 采用 Multiquadric 函数：$\phi(r) = \sqrt{r^2 + c^2}$，其中 $c>0$ 是控制基函数平坦度的平滑参数。该核在复杂环境场的平滑空间模式捕获方面表现优异。[^src-ctenet]

## 在 CTENet 中的应用

CTENet 使用 RBF 插值将稀疏分布的空气质量监测站点（中国 480 个/美国 365 个）的污染物浓度映射到连续欧拉空间网格。消融实验显示，RBF 插值优于最近邻插值（引入突变伪影）和无插值方案（无法有效捕获空间连续性和交互）。[^src-ctenet]

## 优势

相比其他插值方法（最近邻、Kriging、IDW），RBF 具备：灵活性、平滑性、处理不规则分布站点的能力，以及基于核的距离依赖框架适应环境数据的复杂非线性空间变异性。[^src-ctenet]

## 相关页面

- [[ctenet]] — 使用 RBF 插值的模型
- [[air-quality-forecasting]] — 应用领域

[^src-ctenet]: [[source-ctenet]]
