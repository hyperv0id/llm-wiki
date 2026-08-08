---
title: "Tropical Cyclone Precipitation Forecasting"
type: concept
tags:
  - tropical-cyclone
  - precipitation-forecasting
  - extreme-weather
  - deep-learning
created: 2026-07-24
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# Tropical Cyclone Precipitation Forecasting

热带气旋（TC）降水预测是指对伴随 TC 形成、发展和移动的大范围降水事件进行定量预测的任务[^src-tcp]。与常规降水预测的关键区别在于：预测窗口随 TC 中心动态移动，而非固定地理区域[^src-tcp]。

## 与常规降水预测的差异

常规降水预测（[[precipitation-nowcasting|precipitation nowcasting]]）聚焦固定区域，如以雷达覆盖范围为界做 0–6h 外推；而 TC 降水预测的感兴趣区域（10°×10° 窗口）随 TC 的移动而改变，同时区域内的环境因子（如地形、海温）也在变化[^src-tcp]。这意味着常规降水预测方法（U-Net、ConvLSTM、GAN）简单迁移到 TC 任务时效果不佳——[[tcp-diffusion|TCP-Diffusion]] 实验显示，Persistence 基线在这些方法上的表现反而更好，说明它们未能捕捉 TC 特有的降水动力学[^src-tcp]。

## 数据源

TC 降水预测需整合多种异构数据[^src-tcp]：

- **降水观测**：MSWEP（Multi-Source Weighted-Ensemble Precipitation），3h 分辨率 0.1°
- **环境变量**：ERA5 地表数据（2m 温度、SST、MSLP、地形）和气压层数据（温度、比湿、U/V 风分量、位势高度），覆盖 200/600/850/925 hPa
- **TC 属性**：IBTrACS 中的强度、移动速度、月份和轨迹位置
- **NWP 预测**：ERA5-IFS 未来预测数据（ECMWF-IFS 仅用于对比）

## 现状

截至 2025 年，绝大多数 TC 预测研究聚焦于路径和强度，TC 降水被严重忽视[^src-tcp]。[[tcp-diffusion|TCP-Diffusion]]（ICML 2025）是论文自称（To our knowledge）首个基于 DL 的全球 TC 降水预测工作，12h 预测时效内在中等和强降水预测上超越 ECMWF-IFS。TC 形成期和消散期仍是性能瓶颈[^src-tcp]。

[^src-tcp]: [[source-tcp]]
