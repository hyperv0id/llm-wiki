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
**发表**：CIKM 2024（DOI 10.1145/3627673.3679989），arXiv:2309.15169v2
**代码**：github.com/jsun57/STMAE；本地全文：`raw/st-mae.txt`

## 问题与核心论点

交通基准数据仅覆盖数月、限于特定地点，时空模型易 overfitting、对数据缺失不稳[^src-st-mae]。论文提出 STMAE——即插即用的生成式 SSL 框架：继承 MAE（CVPR 2022）的"掩码-重建"原则并迁移到时空交通数据，复用现有 backbone（DCRNN/AGCRN/MTGNN）的 encoder 与 predictor，无需 STGCL 那类手工数据增强[^src-st-mae]。

## 方法机制：双掩码

- **Spatial masking**：掩码单元是路径而非单条边：biased random walk（受 node2vec 启发，融合 BFS/DFS，超参 $p,q$ 控制动态）生成路径，按掩码率 $p_s$ 选出 $|E|\cdot p_s$ 条待掩边并置零邻接矩阵[^src-st-mae]。
- **Temporal masking**：交通数据信息密度低，稀疏单点掩码易被插值恢复；故将 $X$ 分成 $P$ 个长 $L$ 的不重叠 patch，按 Bernoulli($p_t$) 掩码，被掩 patch 换为共享可学习 mask token[^src-st-mae]。
- 两个轻量 decoder 分别重建数据（线性层）与结构（线性+Sigmoid）；损失 $L_{pretrain}=\lambda L_A+L_X$，仅在被掩部分计算。微调丢弃 decoder，encoder 接回 backbone predictor，用完整数据优化 $L_{pred}$[^src-st-mae]。
- 超参网格搜索：patch 长度 $L\in\{2,3\}$、$p,q,\lambda\in\{0.5,1,2,4\}$、掩码率 20%–80%。encoder/predictor 直接取自 backbone，不调深度；预训练 100 epochs + 微调 100 epochs[^src-st-mae]。

## 实验证据

PEMS03/04/07/08（6:2:2，12 步预测 12 步），对照对比式 SSL 的 STGCL（[[source-stgcl]]），MAE/MAPE/RMSE（Table 1）：
- AGCRN：PEMS04 MAE 19.39→19.23（STGCL_A）→19.05（STMAE_A）；PEMS03/07/08 MAE 15.47→15.09 / 20.64→20.13 / 15.65→15.01[^src-st-mae]。
- MTGNN：PEMS03 MAE 14.94→14.84、MAPE 16.02→14.15；PEMS08 MAE 15.44→15.03[^src-st-mae]。
- DCRNN：PEMS08 MAE 16.63→16.36；PEMS04 MAE 21.48→21.20[^src-st-mae]。
- **正文与表格冲突**：正文断言 STMAE "always outperforms STGCL"，但 Table 1 DCRNN 块 PEMS03 MAE 15.76（base）/15.64（STGCL_D）/15.74（STMAE_D）——该场景 STGCL 更优[^src-st-mae]。
- Table 2 消融（STMAE_A，PEMS04/PEMS08，消融表非基线对照）：Base 19.39/15.65，去时间掩码 19.27/15.41，去空间掩码 19.27/15.26，均匀时空掩码 19.11/15.09，完整双掩码 19.05/15.01——联合双掩码最优[^src-st-mae]。
- 掩码率敏感性（AGCRN，仅 PEMS04/08）：时间掩码两数据集均最优 30%；空间掩码 PEMS08 最优 70%、PEMS04 最优 30%，归因于 PEMS08 路网更致密（0.01 vs 0.004）；过高过低均变差[^src-st-mae]。

## 范围与局限

实验仅覆盖 PEMS 四基准、三个 backbone（页面评述）[^src-st-mae]。预训练+微调共 200 epochs 为论文设置；原文未讨论相对直接训练的额外开销[^src-st-mae]。

## 与相关页面

- [[source-mae]]：STMAE 继承其掩码重建范式，把 mask token 从图像 patch 迁移到时间 patch（页面评述）[^src-st-mae]。
- [[source-2312-00516-std-mae]]（IJCAI 2024）：**不同论文**。STD-MAE 是独立模型（S-MAE/T-MAE 双 Transformer 分别沿空间/时间预训练）；STMAE 是包裹任意 backbone 的框架（页面评述）[^src-st-mae]。
- STEP（Shao et al., KDD 2022）：同用掩码机制，但依赖专用 Transformer 编码长期时序；STMAE 定位为对任意 backbone 通用的 enhancer（论文 Related Work 转述）[^src-st-mae]。

[^src-st-mae]: [[source-st-mae]]
