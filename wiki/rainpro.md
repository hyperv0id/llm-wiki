---
title: "RainPro"
type: entity
tags:
  - precipitation-nowcasting
  - probabilistic-forecasting
  - deep-learning
  - weather
  - multi-source-fusion
created: 2026-07-16
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# RainPro

RainPro 是由 Aarhus University 和 Cordulus 开发的概率降水预报模型系列，ICLR 2026 发表。其核心设计哲学是在有限计算资源下实现高效、准确的多源数据融合概率降水预报[^src-rainpro]。

## 模型家族

### RainPro-8

旗舰模型，目标为欧洲 8 小时高分辨率（2km/px，10min 间隔）概率降水预报。整合雷达、卫星、NWP（GFS）和地形数据，36.7M 参数，单次前向传播同时输出 48 个预报时效的概率图[^src-rainpro]。

关键架构选择：U-Net + MaxViT 骨干、有序一致性损失（Ordinal Consistent Loss）、时效权重衰减（exponential decay α=10）、Space-to-Depth 下采样。训练于 NVIDIA H100，约 13 小时完成[^src-rainpro]。

### RainPro-8R

RainPro-8 的纯雷达变体，用于消融实验分离多源数据的贡献。仅使用雷达输入，其他架构和训练设置不变[^src-rainpro]。

### RainPro-2R

为 SEVIR 基准（2 小时雷达预报）适配的轻量变体。去除多源数据和空间上下文，仅用雷达帧做 5→20 帧预测，保留有序一致性损失和单次预测策略。在 CSI/HSS 像素级指标上超越所有确定性和生成式 SOTA，在 CRPS 和 FSS 上超越 DiffCast 且快 13×[^src-rainpro]。

## 与 MetNet 的关系

RainPro 的设计受 [[metnet|MetNet]] 系列启发，但做出关键差异选择：有序一致性损失替代交叉熵、单次预测替代逐时效条件化、大幅减少参数（36.7M vs 227M）。RainPro-8 以不到 20% 的训练参数量在欧洲实现可比或更优性能[^src-rainpro]。

[^src-rainpro]: [[source-rainpro]]
