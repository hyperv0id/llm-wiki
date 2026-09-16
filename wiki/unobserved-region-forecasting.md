---
title: "Unobserved-Region Forecasting"
type: concept
tags:
  - traffic-forecasting
  - generalization
  - kriging
  - extrapolation
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Unobserved-Region Forecasting（无观测区域交通预测）

预测**没有任何传感器观测的连续区域**（与观测区相邻或为其包围）的未来交通状态[^src-gencast]。

## 与相邻任务的区别

| 任务 | 未知点的空间形态 | 代表方法 | 失效点 |
|---|---|---|---|
| 时空克里金 | 零星散点 | IGNNK、INCREASE、[[ustd\|USTD]] | 依赖局部邻域消息传递 |
| 时空外推 | 零星散点 | STGNP、KITS | 依赖已知网络拓扑或预设节点密度 |
| 无观测区域 | 大面积连续 | STSM、[[source-gencast\|GenCast]] | 局部近邻断裂，需全局先验 |

前两类假设未知点处于已知点的局部稠密邻域内；连续片区内部没有活跃节点，图消息传递拿不到观测[^src-gencast]。

## 难点

1. 局部信息流阻断：GCN/GAT 的连边扩散在无节点区域内无信号可传[^src-gencast]。
2. 静态辅助特征无法表达动态：POI 与坐标不反映天气、潮汐流动、偶发事件带来的状态转移[^src-gencast]。
3. 局部过拟合：模型记住特定位置的局部波动，直接迁移到无观测区域造成负迁移[^src-gencast]。

## 已有做法

**掩码子图自监督**：训练时从观测图切除连续子图模拟无观测区，生成伪观测并以图级对比损失约束两种视图的表征一致[^src-gencast]。

**物理约束**：无观测区没有交通数值，但仍有守恒律与速度-密度关系可用；把 [[lwr-traffic-pde|LWR]] 残差作为软约束损失，需自动微分求时空偏导[^src-gencast]。

**外部动态信号**：气象重分析网格（ERA5-Land）覆盖无传感器区域，按最近邻对齐后可作为环境背景输入[^src-gencast]。

**局部特征解耦**：软聚类到潜在空间组并用熵正则化锐化归属，保留跨区域共享的组级模式[^src-gencast]。

## 相关页面

- [[traffic-forecasting|Traffic Forecasting]]
- [[source-gencast|GenCast]] — 上述四点的组合实现
- [[lwr-traffic-pde|LWR 交通流偏微分方程]]
- [[differentiable-spatial-embedding|可微空间嵌入]]
- [[stunet|STUNet]] — 跨网络零样本，问题形态不同（有观测、换路网）
- [[craft|CRAFT]] — 跨城市流量生成，无观测但目标是生成而非预测

[^src-gencast]: [[source-gencast]]
