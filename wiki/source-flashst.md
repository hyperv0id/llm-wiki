---
title: "FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction"
type: source-summary
tags:
  - spatio-temporal
  - prompt-tuning
  - traffic-prediction
  - distribution-shift
  - in-context-learning
  - transfer-learning
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction

**Authors**: Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang (SCUT, HKU, Pazhou Lab)
**Venue**: ICML 2024 (PMLR 235, Vienna), arXiv:2405.17898
**Code**: https://github.com/HKUDS/FlashST

## 核心问题

端到端模型在数据集 A 训练后直接用于分布不同的数据集 B，时空特征差异致性能退化（Figure 1）。解法是模型无关（model-agnostic）的 prompt-tuning：预训练全部参数后冻结下游模型，仅调轻量 prompt 网络适配新数据。

## 方法机制

两阶段。**预训练**：在 PEMS03/04/07/08（加州交通流量，358/307/883/170 区域，16,992–26,208 时间步，Table 1）交替训练 300 epochs，batch 64；**Prompt-tuning**：冻结下游模型 g̃(·) 仅训练 prompt 网络 20 epochs。d/dt/dr 均为 32，时间/空间编码器各 2 层。组件：

- **时空上下文蒸馏**：时间侧拼接 hour-of-day 与 day-of-week 嵌入 M_t=CAT[z^(d)·e1, z^(w)·e2]；空间侧对归一化 Laplacian 特征分解，取 dr 个最小非平凡特征向量作结构属性 C∈R^(R×dr)，MLP 映射后拼接 Ē=CAT[E,M,C̄]。
- **时空依赖编码**：门控时间编码器 H=(W2σ(W1Ē+b1)+b2)+Ē；空间侧图卷积 S=σ(A·H·W3)+H，残差缓解多层 GNN 过平滑。

**统一分布映射**：两阶段共用损失 L=Lr+λL_Uni，Lr 为 MAE 回归损失，L_Uni 为 InfoNCE 均匀性损失（cos 相似度、温度 τ），推动 prompt 嵌入均匀分布以对齐两阶段数据；预训练阶段 f_Prompt 与 g 均可训练，最优 τ=0.3、λ=1.0。

## 实验证据

目标数据集（6:2:2）：PEMS07(M) 洛杉矶车速、CA-D5 加州流量、ChengDu-DIDI 成都流量、NYC Citi Bike 单车需求，13 基线。Table 3 各列最优基线不同，逐列核对：PEMS07(M) MAE 2.59 vs GWN 2.67、CA-D5 13.26 vs GWN 13.63、ChengDu-DIDI 2.31 vs MTGNN 2.33、NYC 1.79 vs GWN 1.81；PEMS07(M) MAPE 6.54% vs GWN 6.62%。

- **模型无关性**（Table 4，STGCN/GWN/MTGNN/PDFormer）：直接迁移严重退化（PEMS07(M) MAE 8.07/7.41/5.91），全参微调 3.18/2.69/2.62，FlashST 2.68/2.67/2.59。
- **效率**（Table 5）：训练时间降 20%–80%（GWN 1042s→222s、STGCN 411s→190s、MTGNN 962s→230s、PDFormer 7524s→1220s）；prompt-tuning 仅 20 epochs，对照上限 100 epochs、早停 25。
- **消融**：-TC/-SC、-TE/-SE、-Uni 均致下降；r/BN 以 BatchNorm 替代均匀性损失，因缺预训练与下游数据的显式分布联系而失效。

## 局限性

页面评述：需在目标数据训练 20 epochs，是参数高效适配、非零样本；且论文写明 prompt-tuning“carried out on the test dataset”（L593-596），适配阶段直接用目标集的测试划分微调，复现时需注意该设定；预训练仅交通流量，跨语义迁移依赖分布对齐；空间上下文依赖每张图的邻接矩阵与 Laplacian 分解。

## 关联

- [[source-opencity]]：同一一作（Zhonghang Li/HKUDS）的后继，从适配骨干演进为直接预训练基础模型（页面评述）。
- [[source-unist]]：同为"预训练+适配"路线，UniST 用 masked autoencoder，FlashST 走模型无关 prompt-tuning 与分布对齐（页面评述）。
- [[source-st-ssdl]]：均匀性损失建立在对比学习 alignment-uniformity 理论（Wang & Isola, ICML 2020，论文引用）之上（页面评述）。

[^src-flashst]: [[source-flashst]]
