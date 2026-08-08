---
title: "Digital Typhoon Dataset"
type: entity
tags:
  - dataset
  - satellite
  - typhoon
  - cyclone
  - time-series
  - climate
created: 2025-07-14
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# Digital Typhoon Dataset

Digital Typhoon 是目前最具代表性且规模最大的开源台风卫星图像数据集，由日本国立情报学研究所（NII）Kitamoto 团队构建[^src-pipe]。

## 数据集规格

| 属性 | 值 |
|------|-----|
| 时间覆盖 | 1978-2023（至今） |
| 时间分辨率 | 1 小时 |
| 目标卫星 | Himawari |
| 空间覆盖 | 西北太平洋盆地 |
| 空间分辨率 | 5 km |
| 图像尺寸 | 512×512 像素（台风中心 1250km 范围） |
| 光谱覆盖 | 红外（其他波段可在网站获取） |
| 地图投影 | 方位角等面积投影 |
| 标定 | 重新标定 |
| 数据格式 | HDF5 |
| 最佳路径数据 | 日本气象厅（JMA） |
| 数据浏览 | [Digital Typhoon 网站](http://agora.ex.nii.ac.jp/digital-typhoon/) |

## 在 PIPE 中的使用

PIPE 将该数据集的卫星图像与最佳路径时间序列（经纬度、中心气压）配对，构造多模态时间序列预测任务：用过去 $H$ 小时的时间序列和卫星图像预测未来 $F$ 小时的位置和强度[^src-pipe]。

## 相关数据集

- 澳大利亚区域扩展版本 [^src-pipe]：PIPE 用于跨区域泛化实验（零样本迁移）

## 相关页面

- [[pipe]] — PIPE 台风预测模型
- [[source-pipe]] — 论文摘要

[^src-pipe]: [[source-pipe]]
