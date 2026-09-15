---
title: "Revealing the Power of Masked Autoencoders in Traffic Forecasting (STMAE)"
type: source-summary
tags:
  - spatio-temporal
  - traffic-prediction
  - masked-autoencoder
  - self-supervised-learning
  - pretraining
  - plug-and-play
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# STMAE: Revealing the Power of Masked Autoencoders in Traffic Forecasting

**作者**：Jiarui Sun, Yujie Fan, Chin-Chia Michael Yeh, Wei Zhang, Girish Chowdhary（UIUC + Visa Research）
**发表**：CIKM 2024（DOI 10.1145/3627673.3679989），arXiv:2309.15169
**代码**：github.com/jsun57/STMAE；本地全文：`downloads/st-mae.txt`

## 核心论点

交通数据集只覆盖数月、单一城市，时空模型易 overfitting 且对传感器缺失不稳定。论文提出 STMAE——即插即用的生成式 SSL 框架：继承 [[source-mae|原始 MAE]]（CVPR 2022）"掩码-重建"原则并迁移到时空交通数据，复用现有 backbone（DCRNN/AGCRN/MTGNN）的 encoder 与 predictor，无需数据增强[^src-st-mae]。

## 方法机制：双掩码策略

预训练阶段对图 $G$ 和数据 $X$ 施加双掩码（dual-masking）[^src-st-mae]：
- **Spatial masking（biased random walk-based）**：掩码单元是路径——biased random walker（融合 BFS/DFS，超参 $p,q$）从根节点集生成路径，按比例选出待掩边集 $E_{mask}$ 并在邻接矩阵置零，迫使 encoder 推断被隐藏的图结构。
- **Temporal masking（patch-based）**：交通数据信息密度低，稀疏单点掩码易被插值恢复；故将 $X$ 分成 $P$ 个长 $L$ 的不重叠 patch，按 Bernoulli($p_t$) 掩码，被掩 patch 换为共享可学习 mask token。
- 两个轻量 decoder 分别重建数据 $\hat{X}$ 与结构 $\hat{A}$；损失 $L_{pretrain} = \lambda L_A + L_X$：$L_A$ 为被掩边交叉熵，$L_X$ 为被掩 patch 的 MAE 回归，仅在被掩部分计算。
- 微调时丢弃两个 decoder，encoder 接回原 backbone predictor，用完整数据优化 $L_{pred}$；encoder 层数 $L\in\{2,3\}$，$\lambda\in\{0.5,1,2,4\}$，掩码比例 20%–80% 网格搜索。

## 实验证据

PEMS03/04/07/08（时间序 6:2:2，12 步预测 12 步），指标 MAE/MAPE/RMSE，对照对比式 SSL 的 STGCL（[[source-stgcl]]）[^src-st-mae]：
- AGCRN backbone：PEMS04 MAE 19.39→19.05（STGCL 仅 19.27）；PEMS08 15.65→15.01，MAPE 10.33→9.79；PEMS03 15.47→15.09；PEMS07 20.64→20.13。
- MTGNN backbone：PEMS03 MAE 14.94→14.84，MAPE 16.02→14.15，RMSE 25.29→24.95；PEMS04 19.02→18.87；PEMS08 15.44→15.03，MAPE 10.35→9.82。
- DCRNN backbone：PEMS04 21.48→21.20；PEMS08 16.63→16.36。
- 消融（PEMS04，AGCRN）：去时间掩码 19.27、去空间掩码 19.27、均匀时空掩码 19.11、完整双掩码 19.05——联合双掩码最优。
- 掩码比例敏感性：时间掩码 20%–30% 在四个数据集最优；空间掩码在 PEMS08 最优 70%，其余 20%–30%（路网结构更致密）。

## 局限性

- 仅在 PEMS 四个基准、三个 backbone 上验证，未覆盖跨城市泛化[^src-st-mae]。
- 预训练+微调共 200 epochs，训练开销高于直接训练 backbone[^src-st-mae]。

## 与相关页面

- [[source-mae]]：STMAE 继承其 encoder-decoder 掩码重建范式，把 mask token 从图像 patch 迁移到时间 patch。
- [[source-2312-00516-std-mae]]（IJCAI 2024）：**不同论文**。STD-MAE 是独立模型（两个解耦的 S-MAE/T-MAE Transformer 沿空间/时间预训练）；STMAE 是包裹任意 backbone 的框架，空间掩码用 biased random walk 作用于邻接矩阵。
- [[source-stgcl]]：对比式 SSL 基线，STMAE 在所有场景 MAE 均优于 STGCL 且无需手工数据增强。
- STEP（Jiang et al., 2023）：同用掩码自编码，但依赖专用 Transformer 编码长期时序；STMAE 对任意 backbone 通用。

[^src-st-mae]: [[source-st-mae]]
