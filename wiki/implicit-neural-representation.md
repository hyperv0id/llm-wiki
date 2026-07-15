---
title: "Implicit Neural Representation"
type: technique
tags:
  - neural-rendering
  - implicit-representation
  - continuous-function
  - coordinate-based
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Implicit Neural Representation (INR)

Implicit Neural Representation（INR，隐式神经表示）将信号编码为连续坐标到值的映射 $f_\theta: \mathbb{R}^2 \to \mathbb{R}^C$，将离散信号（如图像）表示为可微的连续函数。SIREN（Sitzmann et al., 2020）使用周期激活函数（$\sin$）使 INR 能够精确建模高频细节，奠定了 INR 的基础[^src-qcgs]。

## 核心特性

**分辨率无关**：INR 不依赖固定网格，可在任意坐标处查询——这是其最关键的属性。渲染 $H \times W$ 图像需在所有像素中心 $\mathbf{x} \in \Omega$ 处计算 $f_\theta$，复杂度 $\mathcal{O}(HW)$。

**可微性**：INR 对坐标可微，可通过梯度下降端到端优化。

**无限分辨率 vs 密集查询**：分辨率无关是优势，但密集查询使高分辨率合成变慢——这正是 [[gaussian-splatting|Gaussian Splatting]] 选择性渲染策略所解决的核心瓶颈[^src-qcgs]。

## 主要应用

- **3D 场景重建**：NeRF 系列（Mildenhall 2021 → Mip-NeRF → NeRF in the Wild → Instant NGP 哈希编码加速）
- **图像压缩与任意尺度超分辨率**：LIIF（Chen 2021）、ITSRN（Yang 2021）、LTE（Lee & Jin 2022）、CiaoSR（Cao 2023）
- **气象领域**：[[qcgs|QCGS]] 使用 5 层 MLP INR（hidden=128，正弦位置编码）作为高斯参数估计器，为每个查询点预测各向异性协方差和振幅，实现分辨率灵活的降水场生成[^src-qcgs]

## INR 在 QCGS 中的作用

QCGS 通过条件 INR 克服了传统 GS 需逐图像优化的限制。INR 以卫星特征为条件，通过交叉注意力预测高斯参数 $\{\sigma_x, \sigma_y, \rho, \alpha\}$，使模型能跨区域、跨季节泛化。同时仅在降雨支持区域查询 INR，避免在非降水区域浪费计算[^src-qcgs]。

[^src-qcgs]: [[source-qcgs]]
