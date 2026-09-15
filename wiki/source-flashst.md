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

交通预测模型在训练与测试数据分布不一致时泛化能力差：直接把在数据集 A 上端到端训练的模型用于数据集 B，因时空特征分布差异导致性能严重退化[^src-flashst]。FlashST 的解法不是训练一个新模型，而是一个与骨干模型无关（model-agnostic）的 prompt-tuning 框架：先在预训练数据上训练全部参数，再冻结下游模型、只微调轻量 prompt 网络来适配新数据[^src-flashst]。

## 方法机制

框架分两阶段[^src-flashst]：

1. **预训练阶段**：在 PEMS03/04/07/08 四个加州交通流量数据集（358/307/883/170 个区域，12,672–26,208 个时间步）上交替训练 300 epochs，优化目标为 MAE 回归损失。
2. **Prompt-tuning 阶段**：冻结下游模型 g̃(·) 的全部参数，仅更新 spatio-temporal prompt network，在目标数据集上只训练 20 epochs，输出 prompt 嵌入 E(U^A_pro)=f_Prompt(X) 注入下游模型[^src-flashst]。

Prompt 网络包含两个组件[^src-flashst]：

- **时空上下文蒸馏**：时间上下文将 time-of-day 与 day-of-week 编码为可学习嵌入；空间上下文对邻接矩阵做 Laplacian 特征值分解，取最小的 dr 个非平凡特征向量作为区域结构属性 C ∈ R^{R×dr}，经 MLP 映射后与原始嵌入拼接（Ē = CAT[E, M, C̄]）。目标数据无需额外外部数据即可获得结构感知的初始时空嵌入。
- **时空依赖编码**：时间编码器用门控机制 H = (W2σ(W1Ē+b1)+b2)+Ē 建模时隙间演化；空间编码器用图卷积消息传递 S = σ(A·H·W3)+H，配合残差连接缓解多层 GNN 过平滑，堆叠 2 层后输出 prompt 嵌入 E_pro。

**统一分布映射机制**：为弥合预训练数据与下游数据的分布差距，引入 InfoNCE 均匀性损失 L_Uni（余弦相似度 + 温度系数 τ），推动 prompt 嵌入在超球面上均匀分布，使不同来源数据的嵌入共享同一分布空间；总损失为回归损失与 L_Uni 的加权和（λ 系数），最优配置 τ=0.3、λ=1.0[^src-flashst]。

## 实验证据

预训练后在四个目标数据集上评估（6:2:2 划分）：PEMS07(M)（洛杉矶车速）、CA-D5（加州流量）、ChengDu-DIDI（成都流量）、NYC Citi Bike（纽约单车需求），对比 13 个基线[^src-flashst]：

- **端到端对比**：以 MTGNN 为下游模型，FlashST 在四个数据集 MAE 全部领先——PEMS07(M) 2.59 vs MTGNN 2.70，CA-D5 13.26 vs 13.90，ChengDu-DIDI 2.31 vs 2.33，NYC Citi Bike 1.79 vs MSDR 1.87（最强基线）。PEMS07(M) 上 MAPE 从 6.81% 降至 6.54%。
- **模型无关性**：同一框架接入 STGCN、GWN、MTGNN、PDFormer 四种骨干均有效。未微调直接迁移性能崩溃（如 STGCN 在 PEMS07(M) MAE 8.07），全参数微调后为 3.18，FlashST prompt-tuning 达 2.68；GWN 上 7.41 → 微调 2.69 → FlashST 2.67。
- **效率**：prompt-tuning 将训练时间较端到端训练降低 20%–80%（GWN 1042s→222s，STGCN 411s→190s，MTGNN 962s→230s，PDFormer 7524s→1220s），且在 20 epochs 内收敛，快于需要 100 epochs（早停 25）的全量微调。
- **消融**：去除时间/空间上下文蒸馏（-TC/-SC）、时间/空间依赖编码器（-TE/-SE）、统一分布映射（-Uni）均导致性能下降；用 BatchNorm 替代均匀性损失（r/BN）会失去预训练与下游数据的分布联系，知识迁移失效。

## 局限性

- prompt-tuning 需要在目标数据集上训练 20 epochs，并非 OpenCity 式的纯 zero-shot，本质是参数高效的少样本适配。
- 预训练仅用交通流量数据，跨数据类型（车速/单车需求）的迁移依赖统一分布映射而非语义理解。
- 空间编码依赖 Laplacian 特征分解，仍需每张图的邻接矩阵。

## 关联

- [[source-opencity]]：同一一作（Zhonghang Li/HKUDS）的后继工作，从"prompt-tuning 适配已有骨干"演进为"直接预训练时空基础模型"；OpenCity 论文中将 FlashST 所代表的 prompt-tuning/微调路线列为零样本基础模型的对照方案。
- [[source-unist]]：同为"预训练 + 下游适配"的时空通用模型，UniST 用 masked autoencoder 预训练 + 微调，FlashST 则强调模型无关的 prompt-tuning 与分布对齐。
- [[source-st-ssdl]]：同属时空自监督/对比学习预训练路线，FlashST 的 InfoNCE 均匀性损失直接建立在对比学习对齐-均匀（alignment-uniformity）理论（Wang & Isola, ICML 2020）之上。

[^src-flashst]: [[source-flashst]]
