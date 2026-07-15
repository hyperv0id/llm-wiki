---
title: "RainPro-8: Efficient Deep Learning Model for Rainfall Probability over 8 Hours"
type: source-summary
tags:
  - precipitation-nowcasting
  - probabilistic-forecasting
  - multi-source-fusion
  - weather
  - deep-learning
created: 2026-07-16
last_updated: 2026-07-16
source_count: 0
confidence: low
status: active
---

# RainPro-8: Efficient Deep Learning Model for 8-Hour Rainfall Probability

Rafael Pablos Sarabia, Jeppe Liborius Sjørup, Joachim Nyborg, Anders Lillevang Vesterholt, Morten Birk, Ira Assent (Aarhus University & Cordulus), ICLR 2026。

## 核心贡献

提出 **RainPro-8**，一个高效的 8 小时概率降水临近预报模型，在欧洲以 2km/px、10 分钟间隔运行，弥补了临近预报（<2h）与中期预报（>1 天）之间的空白。

三项关键创新：
1. **有序一致性损失（Ordinal Consistent Loss）**：通过条件概率公式 $P(R_t \ge \min(I_c)) = P(R_t \ge \min(I_c)|R_t \ge \min(I_{c-1})) \times P_{t,c-1}$ 强制降水强度类别的单调性，使模型在训练时即学习序数关系，而非仅在推理时通过累积概率约束。
2. **单次前向预测（Single-Pass Predictions）**：将所有 48 个预报时效（10min × 8h）编码到通道维度，一次前向传播同时生成，推理速度比逐时效条件化快 48×，且提升时序一致性。
3. **多源异构数据融合**：整合雷达（RainViewer，4km/8km）、卫星（EUMETSAT，8km 11 通道）、NWP（GFS，16km 122 变量）、地形（Copernicus DEM），通过 Space-to-Depth + ResNet 编码器在对应分辨率层级融合，U-Net + MaxViT 骨干仅 36.7M 参数（MetNet-3 的 16%）。

## 关键结果

- 在欧洲 8 小时预报上全面超越 GFS NWP（CSI +65%）、PySTEPS、Earthformer、SimVP 及忠实复现的 MetNet-3*
- CRPS 0.06096、CSI 0.279、FSS 0.537，优于 MetNet-3* 且推理快 48×
- 消融证实：有序一致性损失、单次预测、时效权重衰减、多源数据各自贡献正向增益
- Integrated Gradients 归因：近期高分辨率雷达驱动短时效（<2h），卫星和 GFS 变量在 4h 后逐渐主导
- SEVIR 基准（2h 雷达预报）：简化版 RainPro-2R 在 CSI/HSS 像素级指标上超越所有确定性和生成式基线，CRPS 和 FSS 超越 DiffCast 且推理快 13×

## 局限性

- 单次训练运行（计算约束），虽有三轮种子鲁棒性验证
- 对缺失数据的鲁棒性未显式处理（运行环境中某些输入源可能不可用）
- 降水强度阈值需人工设定，未自动学习
- 欧洲特定训练，其他地区泛化性待验证


