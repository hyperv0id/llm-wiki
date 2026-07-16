---
title: "StormBench"
type: entity
tags:
  - benchmark
  - weather-forecasting
  - nowcasting
  - multi-modal
  - reanalysis
  - era5
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# StormBench

StormBench 是 [[storminsight|StormInsight]] 论文引入的大规模多源临近预报基准，整合了操作化多源观测与 ERA5 再分析数据，覆盖不同气候类型区域[^src-storminsight]。

## 数据构成

### 观测数据

| 区域 | 卫星通道 | 雷达变量 | 站点变量（6 项共有） |
|------|----------|----------|---------------------|
| 美国 | VIS, IR069, IR107, LGHT | VIL（垂直累积液态水） | 温度、气压、风速、风向、湿度、降水 |
| 法国 | VIS, IR039, WV062, IR108 | CR（雷达反射率） | 同上 |

### ERA5 再分析数据

共 **65+ 变量**，分两层[^src-storminsight]：

- **压力层**（925/850/700/500/300 hPa）：散度、位势高度、相对/比湿、温度、水平/垂直风分量（8 变量 × 5 层 = 40 场）
- **单层**：10m/100m 风、2m 温度/露点、海平面气压、热通量、CAPE、CIN、边界层高度、总柱水汽、云底高度等 25+ 变量

### 时空覆盖

- **美国**：基于 SEVIR 事件数据集，CONUS 域覆盖多种柯本气候类型，384km × 384km 固定区域，4 小时时间窗，5 分钟采样间隔（49 帧/事件），涵盖 2017–2019 年
- **法国**：基于 MeteoNet 长期连续序列，西北区（46.25°–51.896°N）和东南区（41.10°–46.25°N），约 550km × 550km，涵盖 2016–2018 年，兼有温带海洋性和地中海气候类型

## 设计原则

StormBench 在气象专家指导下构建[^src-storminsight]：选取与对流初生、组织和演化最紧密相关的变量；统一异构观测系统下的区域一致学习信号；保留物理链接——环境强迫与风暴尺度行为之间的因果关系。

## 任务设置

13 帧（~1 小时）历史输入 → 25 帧（~2 小时）预测目标，所有模态对齐至 5 分钟 cadence 并空间共配准后分批训练[^src-storminsight]。

[^src-storminsight]: [[source-storminsight]]
