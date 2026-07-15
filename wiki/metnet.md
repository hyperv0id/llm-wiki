---
title: "MetNet"
type: entity
tags:
  - precipitation-nowcasting
  - probabilistic-forecasting
  - deep-learning
  - weather
  - google
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: low
status: active
---

# MetNet

MetNet 是由 Google Research 开发的概率降水预报深度学习模型系列，包括 MetNet-1（Sønderby et al., 2020）、MetNet-2（Espeholt et al., 2022, Nature Communications）和 MetNet-3（Andrychowicz et al., 2023），代表了深度降水预报的最先进水平[^src-rainpro]。

## 核心设计

- **概率输出**：使用交叉熵损失在降水强度 bin 上直接预测概率分布，一次前向传播代替集合方法[^src-rainpro]
- **时效条件化（Lead Time Conditioning）**：模型一次仅预测一个时效，训练时按时效采样，推理时需对每个时效分别前向传播[^src-rainpro]
- **多源数据融合**：MetNet-3 整合雷达、卫星、天气站、同化气象状态等多源数据，使用 Transformer 骨干[^src-rainpro]
- **大规模训练**：MetNet-3 拥有 227M 参数，在 512 TPU v3 核心上训练 7 天，仅限美国本土[^src-rainpro]

## 局限性

- 计算成本极高，且代码和训练数据未公开，限制可复现性[^src-rainpro]
- 交叉熵损失忽略降水强度类别间的序数关系，需要推理时后处理保证概率一致性[^src-rainpro]
- 逐时效推理导致 48× 的推理开销，且可能引入时效间时序不一致[^src-rainpro]
- 依赖美国本土高分辨率数据（HRRR 3km），欧洲可用等效数据（GFS 28km）分辨率显著较低[^src-rainpro]

## 与 RainPro 的关系

[[rainpro|RainPro]] 基于 MetNet-3 的核心架构理念进行了多项关键改进，以不到 20% 的参数量在欧洲实现可比或更优性能。RainPro-8 论文包含 MetNet-3*（忠实复现版本）作为直接对照[^src-rainpro]。

[^src-rainpro]: [[source-rainpro]]
