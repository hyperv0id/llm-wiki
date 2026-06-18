---
title: "JSTC: Travel Time Prediction with a Joint Spatial-Temporal Correlation Mechanism"
type: source-summary
tags:
  - travel-time-prediction
  - spatial-temporal
  - attention-mechanism
  - convolutional-network
  - intelligent-transportation
  - od-based
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# JSTC 论文摘要

**JSTC**（Joint Spatial-Temporal Correlation）由大连理工大学 Alfateh M. Tag Elsir 等人发表于 Journal of Advanced Transportation 2022。论文提出了一个统一的深度学习框架，通过联合时空相关机制提升 OD（起点-终点）行程时间预测精度[^src-jstc]。

## 核心贡献

1. **空间稀疏性解决方案**：通过 Geo-hashing 将城市划分为 N×N 网格，结合 K-means 聚类（100 个簇），使模型能够利用邻近行程记录处理稀疏数据。

2. **联合时空相关模块**：设计了空间自注意力模块（SSAM）和残差扩张卷积模块（RDCM），分别捕捉空间关系和时间依赖。

3. **多头注意力特征融合**：采用多头注意力机制学习空间、时间和外部特征对输出的贡献权重，同时支持并行计算加速训练。

## 核心方法

### 空间自注意力模块（SSAM）

对空间特征张量（OD 坐标、簇 ID、网格 ID、距离、速度等）使用 1D 卷积层生成 Query、Key、Value，通过缩放点积注意力计算空间特征间的相关性得分。该模块能够推断空间关系，特别是当历史数据中不存在相同位置的记录时，可以借助邻近区域的信息[^src-jstc]。

### 残差扩张卷积模块（RDCM）

采用三层堆叠的扩张 1D 卷积（dilation rates = {1, 2, 4}），扩大感受野以覆盖整个输入序列的时序依赖关系。残差连接保留原始输入信息，避免梯度消失。输入包括时间特征（星期几、小时、月份、工作日/周末）和辅助空间特征（网格和簇的密度得分）。

### 多头注意力模块（MHAM）

外部特征（天气 10 类和节假日）经全连接层处理后，与时空特征拼接，通过 6 头注意力机制学习各特征对行程时间的贡献权重。最终经线性层输出预测结果。

## 实验结果

### 整体性能

在 NYC（7500 万条记录）、成都（970 万条）、西安（527 万条）三个数据集上，JSTC 在 MAPE 和 MAE 上均显著优于所有对比方法[^src-jstc]：

| 模型 | NYC MAPE | Chengdu MAPE | Xi'an MAPE |
|------|:---------:|:------------:|:----------:|
| LRM | 26.12 | 24.37 | 25.85 |
| XGBoost | 25.39 | 22.59 | 23.37 |
| LightGBM | 22.19 | 21.98 | 21.51 |
| ST-NN | 20.04 | 19.02 | 20.44 |
| TTE-Ensemble | 18.33 | 17.58 | 18.86 |
| FMA-ETA | 15.81 | 15.74 | 16.04 |
| STTNs | 14.38 | 14.25 | 15.74 |
| **JSTC** | **13.14** | **12.08** | **14.13** |

相比最强基线 STTNs，MAPE 分别降低 1.24%、2.17%、1.61%。

### 高峰期表现

在早高峰（7–10 AM）和晚高峰（5–8 PM）时段，JSTC 仍保持最优，MAPE 相比 STTNs 降低约 1.5–2.0 个百分点。

### 消融实验

| 移除模块 | NYC MAPE 增长 | 影响程度 |
|----------|:-------------:|:--------:|
| 移除 SSAM | +8.22 | 最大 |
| 移除 RDCM | +6.95 | 第二大 |
| 移除外部因素 | +4.29 | 显著 |
| 移除 MHAM | +3.67 | 较小 |

SSAM 和 RDCM 对模型影响最大，验证了联合时空相关机制的核心作用。外部因素（天气、节假日）也显著贡献于预测精度。

## 局限性

- 模型复杂度较高，训练时间略长于简单模型（但预测速度与 STTNs 接近）
- 仅使用 OD 坐标的 GPS 信息，未考虑路线路径信息
- 外部因素仅包括天气和节假日，未考虑事件、施工等更多因素

[^src-jstc]: [[source-jstc]]