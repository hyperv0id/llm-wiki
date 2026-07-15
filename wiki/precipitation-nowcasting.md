---
title: "Precipitation Nowcasting"
type: concept
tags:
  - weather-forecasting
  - precipitation
  - nowcasting
  - radar
  - satellite
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Precipitation Nowcasting

降水临近预报（Precipitation Nowcasting）是指对 0–6 小时短时降水进行高分辨率预测的任务。与中期天气预报（1–15 天）不同，临近预报关注的是局部、快速演变的对流系统，对空间分辨率和时效性要求极高[^src-qcgs]。

## 为什么降水特别难

尽管基于 ERA5 的数据驱动全球预报模型（FourCastNet、Pangu-Weather、GraphCast、ClimaX、FuXi、GenCast）已在中程超越 NWP，降水仍然是突出难点。根本原因在于**尺度不匹配**：全球模型运行在数十公里粗分辨率上，而降水关键特征（局部暴雨、对流单体）出现在亚网格尺度、间歇且局部地涌现[^src-qcgs]。

## 方法演进

### 传统方法
- **光流法**：Lucas-Kanade 等对雷达反射率场做外推（如 PySTEPS），预报技巧受限于雷达保真度
- **统计插值**：Barnes 插值、Kriging、最优插值从雨量计点观测构建网格化降水场，但模糊尖锐边界、对站点密度敏感

### 深度学习时代
- **ConvLSTM**（Shi et al., 2015）：开创性工作，将卷积嵌入 LSTM 做时空预报
- **GAN-based**：DGMR（Ravuri et al., 2021, Nature），深度生成式雷达降水预报
- **扩散模型**：PreDiff（Gao et al., 2023）、DiffCast（Yu et al., 2024a）、CasCast（Gong et al., 2024a）、PostCast（Gong et al., 2024b）在短时效实现强性能

### 雷达困境
上述方法几乎全部假设雷达为主要输入。但雷达网络成本高、地理覆盖有限，主要适用于欧美等地区。雷达分辨率固定，无法表征亚尺度过程[^src-qcgs]。

## 无雷达路线

| 路线 | 代表方法 | 优缺点 |
|------|----------|--------|
| 纯卫星 | Sat2Radar (NPM, Park et al. 2025) | 全球覆盖，但偏差大、固定分辨率 |
| 卫星-站点融合 | QCGS (Kim et al., ICLR 2026) | 分辨率灵活、精度高，但依赖 AWS 密度；核心依赖 [[gaussian-splatting|Gaussian Splatting]] 和 [[implicit-neural-representation|INR]] |
| 传统插值 | Kriging, Barnes | 无需训练，但模糊边界 |

## 与极端天气预测的关系

临近预报与 [[extreme-weather-forecasting|极端天气预测]] 高度相关：强降水本身就是高影响极端事件。但临近预报关注短时（<6h）局地降水场重建，极端天气预测关注更广泛的事件类型（洪涝、热浪、闪电等）和更长的预报时效。[[qcgs|QCGS]] 的降水场生成能力可作为极端降水预测的数据同化前端[^src-qcgs]。

[^src-qcgs]: [[source-qcgs]]
