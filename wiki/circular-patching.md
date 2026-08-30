---
title: "Circular Patching"
type: technique
tags:
  - patching
  - spherical-geometry
  - weather-forecasting
  - transformer
created: 2026-07-14
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Circular Patching

**圆形分块（Circular Patching）** 是 [[cirt|CirT]] 提出的按纬度线将球面气象数据分解为等距一维 patch 的分块策略，旨在消除标准 grid patching 的几何失真[^src-cirt]。

## 动机

标准 ViT 式分块将 $H \times W$ 经纬网格按固定角度（如 $3^\circ \times 3^\circ$）切分为矩形 patch。在球面上，这导致[^src-cirt]：

- 高纬度 patch 几何面积远小于低纬度——信息密度不均
- Patch 形状在高纬度区域严重扭曲
- 左右边界的空间连续性被切断

## 方法

给定输入 $X \in \mathbb{R}^{H \times W \times K}$（$H$ 条纬线，$W$ 条经线，$K$ 个变量）[^src-cirt]：

1. **按纬度分解**：将输入切分为 $H$ 个互不重叠的 circular patches $\{X^{(h)}\}_{h=1}^H$，其中 $X^{(h)} \in \mathbb{R}^{W \times K}$ 是第 $h$ 条纬线上所有 $K$ 个变量的值
2. **几何性质**：
   - Patch $X^{(h)}$ 的几何长度：$2\pi R\cos(\lambda_h)$，随纬度变化但整条纬线一致
   - 相邻 patch $X^{(h)}$ 和 $X^{(h+1)}$ 间距固定为 $R\Delta\phi$
   - 每个 patch 具有天然的 $2\pi$ 空间周期性：$X_w = X_{w+W}$
3. **嵌入**：展平并堆叠为 $X^F \in \mathbb{R}^{H \times (W\cdot K)}$，线性投影到隐空间 $E \in \mathbb{R}^{H \times D}$

## 与其他分块策略对比

| 策略 | 方法 | 几何一致性 | 代表模型 |
|:-----|:-----|:----------|:---------|
| Grid Patching | 固定角度矩形分块 | 差（高纬失真） | ClimaX, FourCastNetV2 |
| Cube Patching | 立方体面投影分块 | 中（边界不连续） | PanguWeather |
| Mesh | [[multi-mesh-representation|多分辨率球面网格]] | 好（无投影） | [[graphcast|GraphCast]] |
| **Circular Patching** | **按纬度线分块** | **好（等距+周期）** | **CirT** |

## 与傅里叶变换的协同

Circular patching 的 $2\pi$ 周期性天然适配 [[fourier-self-attention|傅里叶域自注意力]]：DFT 将 circular patch 分解为周期基函数的系数，频域注意力可捕获全局空间交互。Ablation 表明单独使用 circular patching 已有提升，但与傅里叶变换组合后提升显著更大[^src-cirt]。

## 相关页面

- [[cirt]] — CirT 模型
- [[fourier-self-attention]] — 傅里叶域自注意力
- [[spherical-geometry-inductive-bias]] — 球面几何归纳偏置
- [[graphcast]] — mesh 路线代表模型
- [[multi-mesh-representation]] — 多分辨率球面网格
- [[source-cirt]] — CirT 论文

[^src-cirt]: [[source-cirt]]
