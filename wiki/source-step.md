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

**STEP**（KDD '22，中科院计算所 Shao Zezhi 等）提出"预训练增强的时空图神经网络"框架，用掩码自编码预训练模型 TSFormer（Time Series Former）弥补 STGNN 只能吃 12 步（1 小时）短历史输入的缺陷[^src-step]。

## 方法机制

**预训练阶段（TSFormer）**：将历史信号按 patch size $L=12$ 切成 $P$ 个不重叠 patch（METR-LA/PEMS-BAY 取 $P=168$ 即一周，PEMS04 取 $P=336$ 即两周），随机掩码 $r=75\%$ 的 patch，仅在被掩码 patch 上计算 MAE 重建损失。encoder 4 层 Transformer block（hidden $d=96$、4 头），decoder 仅 1 层；可学习位置编码是成功关键——换成 sinusoidal 编码后学不到有效表征。$r=75\%$ 时训练速度 279.3 s/epoch，远低于 $r=20\%$ 的 738.6 s/epoch[^src-step]。

**图结构学习**：沿用 GTS 框架但用 TSFormer 表征替代原始序列相似度。边概率 $\Theta_{ij} = FC(relu(FC(Z_i \| Z_j)))$，其中 $Z_i = relu(FC(H_i)) + G_i$（$H_i$ 为该序列全部 patch 表征拼接，$G_i$ 由整个训练序列卷积得到，静态），用 TSFormer 表征算的 kNN 图 $A^a$ 做交叉熵正则，经 Gumbel-Softmax 重参数化采样得到可微的离散邻接矩阵，$\lambda = 1/\lceil epoch/6 \rceil$ 逐渐衰减以跳出 kNN 约束[^src-step]。

**下游预测器集成**：冻结预训练 encoder，把其 patch 表征 $H^P$ 经 semantic projector（MLP）变换后与 Graph WaveNet 隐层表示相加：$H^{final} = SP(H^P) + H^{gw}$，端到端训练 $L = L_{regression} + \lambda L_{graph}$。框架也可接入 DCRNN 等 seq2seq 类后端[^src-step]。

## 实验结果

METR-LA（207 传感器，34272 样本）、PEMS-BAY（325，52116）、PEMS04（307，16992），5 分钟采样，预测 12 步。METR-LA 上 STEP H3/H6/H12 的 MAE 为 2.61/2.96/3.37（RMSE 4.98/5.97/6.99，MAPE 6.60%/7.96%/9.61%），均显著优于最佳基线 Graph WaveNet（H3 2.69/5.15/6.90%），t-test $p<0.05$；PEMS-BAY H3 MAE 1.26 vs GWNet 1.30；PEMS04 H3 MAE 17.34 vs GWNet 18.15。消融验证：去掉图结构学习（w/o GSL）后段级表征仍是主要增益来源；用原始序列余弦相似度替代 TSFormer 表征算 kNN 正则（w/o reg）性能下降，说明长序列表征提升图质量；STEP-DCRNN 证明框架通用[^src-step]。

## 局限性

- 预训练按数据集分别进行（数据异质：长度、物理含义、时间模式不同），未探索跨数据集泛化
- 下游仍需 STGNN，长历史以冻结段级表征形式间接注入

## 关联页面

- [[source-2312-00516-std-mae]] — STD-MAE（IJCAI-24）沿用"冻结预训练表征增强任意下游预测器"的 STEP 范式，但掩码从纯时间 patch 扩展为空间/时间解耦双掩码，以 GWNet 为预测器在六基准上超越 STEP，且推理比 STEP 加速 22.6%–72.5%
- [[source-opencity]] — 后续交通基础模型沿用 patch 化 + instance normalization 思路做零样本泛化

[^src-step]: [[source-step]]
