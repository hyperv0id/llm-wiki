---
title: "Ordinal Consistent Loss"
type: technique
tags:
  - loss-function
  - probabilistic-forecasting
  - ordinal-classification
  - precipitation-nowcasting
created: 2026-07-16
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Ordinal Consistent Loss

有序一致性损失（Ordinal Consistent Loss）是一种用于有序分类任务的损失函数，由 RainPro-8 论文引入概率降水预报场景，确保预测概率对不同强度类别保持单调性[^src-rainpro]。

## 动机

传统降水概率预报（如 MetNet）使用交叉熵损失独立预测每个强度区间 $P(R_t \in I_c)$，忽略了强度类别的内在序数关系——降雨强度 ≥5mm/h 必然是 ≥1mm/h 的子集。这导致模型可能在推理时出现概率不一致（如 $P(R_t \ge 5.0) > P(R_t \ge 1.0)$），需要后处理修正[^src-rainpro]。

## 公式化

模型不直接预测 $P(R_t \in I_c)$，而是预测条件概率 $P(R_t \ge \min(I_c) \mid R_t \ge \min(I_{c-1}))$，再通过贝叶斯链式法则构建累积概率[^src-rainpro]：

$$P_{t,c} = P(R_t \ge \min(I_c)) = \prod_{j=1}^{c} P(R_t \ge \min(I_j) \mid R_t \ge \min(I_{j-1}))$$

由于每个条件概率因子 $\in [0, 1]$，乘积自然保证 $P_{t,c} \le P_{t,c-1}$ 对所有 $c$ 成立。

损失为每个像素、类别、时效上的加权二元交叉熵（BCE），仅在前一类激活的像素上计算（有序一致性掩码 $R_t(h,w) \ge \min(I_{c-1})$），鼓励模型利用类别序数关系[^src-rainpro]。

## 效果

RainPro-8 消融实验显示，有序一致性损失相比交叉熵损失在 CRPS（0.06096 vs 0.06098）、CSI（0.2791 vs 0.2787）和 FSS（0.5367 vs 0.5357）上均有小幅但一致的提升。三独立种子实验（附录 G）确认提升稳定[^src-rainpro]。

该方法最早由 Fernandes & Cardoso (2018) 在序数图像分割中提出，RainPro-8 首次将其应用于概率降水预报[^src-rainpro]。

[^src-rainpro]: [[source-rainpro]]
