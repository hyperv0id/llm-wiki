---
title: "STEP: Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"
type: source-summary
tags:
  - spatiotemporal-forecasting
  - masked-pre-training
  - gnn
  - self-supervised-learning
  - graph-structure-learning
  - kdd-2022
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# STEP 论文摘要

STGNN 受模型复杂度限制，多数只能吃过去一小时（12 步）的短历史输入，而时空模式需要更长历史才能刻画[^src-step]。**STEP**（KDD '22，中科院计算所 Shao 等）用掩码自编码预训练模型 TSFormer 从数周长历史提取段级表征，作为下游 STGNN 的上下文。

## 方法机制

**预训练（TSFormer）**：序列切成 L=12 的非重叠 patch（METR-LA/PEMS-BAY 取 P=168 即一周，PEMS04 取 P=336 即两周），随机掩码 r=75% 的 patch，仅对被掩码 patch 计算 MAE 重建损失。encoder 仅 4 层 Transformer block（d=96、4 头），decoder 1 层；位置编码用可学习的而非 MAE 的 sinusoidal 版本，论文推测可学习位置编码是 TSFormer 成功的关键（换成 sinusoidal 后学不到有效表征）。掩码率越高 encoder 越快（Figure 5：s/epoch 从 r=20% 的 738.6 降至 r=90% 的 279.3）。

**下游集成**：冻结预训练 encoder，最后 patch 的表征 $H^P$ 经 semantic projector（MLP）与 Graph WaveNet 隐层相加：$H^{final} = SP(H^P) + H^{gw}$，总损失 $L = L_{regression} + \lambda L_{graph}$ 端到端训练。

METR-LA（207 传感器/34272 样本）、PEMS-BAY（325/52116）、PEMS04（307/16992），5 分钟采样。METR-LA 上 STEP 的 H3/H6/H12 MAE 为 2.61/2.96/3.37，对照论文选作 backbone 的 Graph WaveNet（2.69/3.07/3.53；GWNet 并非该列全表最优——GTS H6 3.04、GMAN H12 3.44 更低），全部指标 t-test p<0.05。PEMS-BAY H3 MAE 1.26 vs 1.30，PEMS04 H3 17.34 vs 18.15。消融：w/o GSL 仍有满意性能，段级表征是主要增益来源；w/o reg 用原始序列余弦相似度替换 TSFormer 表征后变差；STEP-DCRNN 验证框架对 seq2seq 类后端通用。

论文按数据集分别预训练，理由是三个数据集在序列长度、物理含义与时间模式上异质——这是设计说明而非论文自认的局限；页面评述认为跨数据集预训练与泛化是未验证的开放问题。长历史仅以冻结段级表征间接注入下游；论文自认难以直观解释 TSFormer 学到了什么。

## 关联页面

- [[source-2312-00516-std-mae]]（页面评述）沿用"冻结预训练表征增强任意下游预测器"的 STEP 范式，但掩码改为空间/时间解耦双掩码，并报告在 GWNet 为预测器时超越 STEP 的结果。

[^src-step]: [[source-step]]
