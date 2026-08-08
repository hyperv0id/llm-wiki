---
title: "Subseasonal-to-Seasonal Forecasting"
type: concept
tags:
  - weather-forecasting
  - s2s
  - climate
  - predictability-desert
  - sea-ice
created: 2026-07-14
last_updated: 2026-08-08
source_count: 3
confidence: medium
status: active
---

# Subseasonal-to-Seasonal (S2S) Forecasting

**次季节到季节（S2S）预测** 指提前 2–6 周预测气象变量的任务，处于中期天气预报（≤15 天）和季节预测（3–6 个月）之间的"可预测性荒漠"[^src-cirt]。

## 为什么难

S2S 时间尺度的根本挑战在于[^src-cirt]：

- **太长**：超过大气初始条件的记忆长度，初始场的混沌发散使逐日预报失效
- **太短**：不足以让海洋、海冰、陆面等缓变地球系统分量对大气产生决定性强迫
- 中期预报靠初始条件，季节预报靠边界强迫，S2S 两者都靠不上

## 方法论演进

**物理方法（NWP）**：ECMWF、UKMO、NCEP、CMA 等机构的耦合模式（如 IFS、GloSea6、CFSv2）通过数值积分热力学和流体方程做 S2S 预测，但普遍存在显著偏差且计算成本极高[^src-cirt]。

**数据驱动方法**：

- **区域方法**：AutoKNN、XGBoost 等在特定区域做 S2S 预测，但无法建模遥相关
- **全球迭代模型**：FourCastNetV2、PanguWeather、GraphCast 等中期预报模型通过自回归迭代扩展到 S2S 尺度，但累积误差严重
- **全球直接模型**：ClimaX 和 CirT 直接预测 S2S 时间窗口的平均值，避免迭代误差
- **几何感知模型**：[[cirt|CirT]]（ICLR 2025）首次将球面几何归纳偏置引入 S2S Transformer，在高纬度区域改善尤为显著[^src-cirt]
- **掩码生成模型**：[[omnicast|OmniCast]]（NeurIPS 2025）将掩码生成建模应用于 S2S 预测（论文提出的新应用），跨时空联合并行解码避免累积误差，论文报告其在 S2S 尺度达到 SOTA，10–20× 快于 GenCast[^src-omnicast]
- **海冰多粒度预测**：[[sifusion|SIFusion]]（NeurIPS 2025）首次将多时间粒度（日/周/月）统一建模引入北极海冰 S2S 预测，通过 granularity variate attention 捕获跨粒度相关性，仅用 SIC 数据超越所有单粒度 baseline[^src-sifusion]

## 与其他预测尺度的关系

| 尺度 | 时间范围 | 核心驱动力 | 代表方法 |
|:-----|:---------|:-----------|:---------|
| 中期预报 | ≤15 天 | 初始条件 | PanguWeather, GraphCast |
| **S2S** | **2-6 周** | **过渡区** | **OmniCast, CirT, ClimaX, SIFusion** |
| 季节预测 | 3-6 月 | 边界强迫（海洋、海冰等） | 耦合气候模式 |

## 相关页面

- [[omnicast]] — OmniCast 掩码潜扩散 S2S 预测模型（NeurIPS 2025）
- [[source-cirt]] — CirT 论文摘要
- [[weather-foundation-model]] — 天气基础模型范式
- [[extreme-weather-forecasting]] — 极端天气预测
- [[spherical-geometry-inductive-bias]] — 球面几何偏置
- [[sifusion]] — SIFusion 多粒度海冰 S2S 预测
- [[sea-ice-concentration-forecasting]] — 海冰密集度预测
[^src-cirt]: [[source-cirt]]
[^src-omnicast]: [[source-omnicast]]
[^src-sifusion]: [[source-sifusion]]
