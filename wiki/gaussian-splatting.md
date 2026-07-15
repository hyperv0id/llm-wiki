---
title: "Gaussian Splatting"
type: technique
tags:
  - gaussian-splatting
  - neural-rendering
  - 3d-vision
  - implicit-representation
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Gaussian Splatting

Gaussian Splatting (GS) 是一种将场景表示为可学习各向异性高斯原语集合的渲染技术。3D Gaussian Splatting (3DGS) 由 Kerbl et al. (2023) 提出，通过用高斯原语替代 NeRF 的体积采样并避免冗余渲染，实现了实时新视角合成。GS 此后被扩展到 4D 动态场景、2D 图像压缩与超分辨率等任务[^src-qcgs]。

## 3DGS → 2DGS

3DGS 中每个原语拥有 3D 中心 $\mu_k \in \mathbb{R}^3$、协方差 $\Sigma_k \in \mathbb{S}^3_{++}$、不透明度 $\alpha_k$ 和颜色 $c_k$，渲染需投影到像平面、雅可比线性化协方差、深度排序前向合成。

2D Gaussian Splatting (2DGS) 移除了几何相关元素，直接在图像平面操作。每个原语简化为：

$$\boldsymbol{\mu}_i \in \mathbb{R}^2, \quad \Sigma_i \in \mathbb{S}^2_{++}, \quad \alpha_i \in \mathbb{R}$$

渲染值：$I(\mathbf{x}) = \sum_{i=1}^K \alpha_i \exp\!\Big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^\top \Sigma_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\Big)$

2DGS 保留了分辨率无关渲染的优势，同时更简单、计算更廉价，特别适合表示尖锐、局部化的降水场结构，如 [[precipitation-nowcasting|降水临近预报]] 中的对流单体[^src-qcgs]。

## 与传统高斯加权插值的关系

气象学中长期使用的客观分析方法（Barnes 插值、Kriging）对点观测应用固定各向同性高斯核进行加权求和。这恰好是 GS 的特例——GS 进一步允许各向异性协方差和可学习振幅[^src-qcgs]。这一等价性是 [[qcgs|QCGS]] 方法的核心动机。

## 2DGS 在图像领域的扩展

GaussianImage、Image-GS、LIG 等将 2DGS 应用于图像压缩和表示，基于梯度或频率内容自适应分配高斯。GaussianSR、ContinuousSR、GSASR 引入核库和前馈预测以实现可扩展性和泛化。但这些方法仍为图像专用，QCGS 首次将 2DGS 扩展到降水场生成[^src-qcgs]。

## 与 INR 的关系

[[implicit-neural-representation|INR]] 将信号编码为连续坐标到值的映射，具有分辨率无关的优势，但需要查询所有坐标且缺乏显式空间结构。GS 通过显式的高斯原语集合既保留了分辨率无关性，又支持选择性渲染（仅评估降水支持区域），在效率上优于纯 INR 方法[^src-qcgs]。

[^src-qcgs]: [[source-qcgs]]
