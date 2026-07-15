---
title: "Spherical Geometry Inductive Bias"
type: concept
tags:
  - spherical-geometry
  - weather-forecasting
  - inductive-bias
  - geometric-distortion
created: 2026-07-14
last_updated: 2026-07-15
source_count: 1
confidence: medium
status: active
---

# Spherical Geometry Inductive Bias

**球面几何归纳偏置** 指在气象/气候机器学习模型中显式编码地球球面几何特性（而非将经纬网格视为平面图像）的设计理念[^src-cirt]。

## 问题：平面投影的失真

全球气象数据定义在经纬网格（graticule）上，这是一个球面坐标系统。大多数数据驱动模型（FourCastNetV2、PanguWeather、ClimaX）将其视为 3D 张量 $X \in \mathbb{R}^{H \times W \times K}$，等价于平面图像。这引入两类严重失真[^src-cirt]：

**1. 面积失真**：等角距分块（如 $3^\circ \times 3^\circ$）在球面上对应不等的几何面积。低纬度 patch 面积显著大于高纬度 patch，导致信息分布不均匀。在高纬度尤其严重——同一 patch 在极地区域相比赤道区域可缩小数倍。

**2. 空间关系失真**：平面视图的左右边界在球面上是连接的，但平面处理将它们视为分离的。纬度方向的周期性（$X_w = X_{w+W}$）也被忽略。

## 解决方案

[[cirt|CirT]] 提出了两种互补的几何偏置设计[^src-cirt]：

- **[[circular-patching|圆形分块]]**：按纬度分块而非固定角度分块。每条纬线作为一个 patch，patch 几何长度由 $2\pi R\cos(\lambda_h)$ 确定，相邻 patch 等距。各 patch 形状一致，消除了面积失真。
- **[[fourier-self-attention|傅里叶域自注意力]]**：利用 circular patch 的天然 $2\pi$ 周期性，在频域执行注意力操作，显式编码空间周期结构。

## 其他编码球面偏置的尝试

- **GraphCast**：用多分辨率 mesh 在球面上建模，通过消息传递聚合局部信息。不显式编码空间周期性[^src-cirt]。
- **PanguWeather**：设计地球特定的位置偏置（earth-specific positional bias），将相对坐标编码为可学习参数注入注意力计算。但仍用 cube patching，隐式学习几何[^src-cirt]。
- **CaFA**：在球面上使用分解注意力（factorized attention），直接在球面坐标下操作[^src-cirt]。
- **SFNO**：基于傅里叶神经算子（FNO）框架的球面扩展，在球面谱域进行运算[^src-cirt]。

## 相关页面

- [[cirt]] — CirT 模型
- [[circular-patching]] — 圆形分块技术
- [[fourier-self-attention]] — 傅里叶域自注意力
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测
- [[weather-foundation-model]] — 天气基础模型

[^src-cirt]: [[source-cirt]]