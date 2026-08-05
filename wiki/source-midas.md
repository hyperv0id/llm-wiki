---
title: "MIDAS: Mutual Information Disentanglement with Uncertainty-Aware Fusion for Incomplete Multimodal Sentiment Analysis"
type: source-summary
tags:
  - multimodal-sentiment-analysis
  - incomplete-multimodal-learning
  - mutual-information
  - disentanglement
  - uncertainty-aware-fusion
  - tpami-2026
created: 2026-08-05
last_updated: 2026-08-05
source_count: 0
confidence: medium
status: active
---

# MIDAS: Mutual Information Disentanglement with Uncertainty-Aware Fusion for Incomplete Multimodal Sentiment Analysis

**Wen, Zhou, Li, Gao, Wen, Tao & Li (2026)，IEEE TPAMI（accepted，DOI 10.1109/TPAMI.2026.3713694）**

完整论文（14 页）：`raw/Wen 等 - 2026 - MIDAS Mutual Information Disentanglement With Uncertainty-Aware Fusion for Incomplete Multimodal Se.pdf`。代码：https://github.com/ultramarineX/MIDAS 。作者主要来自北京邮电大学与清华大学（第一作者 Yuhua Wen，通讯作者 Ya Li）。

## 核心论题

论文提出 MIDAS，面向不完全多模态情感分析（Multimodal Sentiment Analysis, MSA）的统一框架。论文认为现实中的多模态输入常因传感器故障、异步、带宽限制或隐私约束而缺失或损坏，而现有两类方法都有缺陷：数据填充类（TFR-Net、NIAT、CIF-MMIN、EMT 等）计算开销大且重构误差累积；协调表示类（ALMT、LNLN、P-RMF 等）依赖启发式协调约束，难以学习一致的跨模态特征。论文将不完全 MSA 类比为域泛化问题——不同缺失模式造成分布偏移——并主张用解耦表示学习重构被损坏的多模态信息。

## 方法

MIDAS 由三个组件构成：

1. **变分建模（Variational Modeling, VM）**：每个模态 $X_m$ 用两个多元高斯潜变量建模，分解为共享因子 $Z^s_m \sim \mathcal{N}(\mu^s_m, \sigma^s_m I)$ 与独有因子 $Z^e_m \sim \mathcal{N}(\mu^e_m, \sigma^e_m I)$，重参数化采样。方差 $\sigma$ 量化该模态的 aleatoric 不确定性，均值 $\mu$ 作为稳定表示（推理时使用）。
2. **互信息 minimax（MI Minimax, MIM）**：最小化共享与独有空间之间的互信息 $I(Z^s_m; Z^e_m)$ 实现解耦——利用 interaction information 分解为三项，假设条件独立后推导出由 KL 散度与重构似然组成的变分上界；同时最大化跨模态共享空间之间的互信息 $I(Z^s_{m_1}; Z^s_{m_2})$ 增强语义对齐——用 Deep InfoMax 的 JSD 估计器，并以独有特征构造硬负样本。另设辅助二元分类弱监督损失与 MSE 重构损失，防止共享空间崩溃并保留语义内容。
3. **不确定性感知融合（Uncertainty-Aware Fusion, UAF）**：把三个模态的共享/独有特征堆叠为 6 个 token 的序列，用后验方差的微分熵 $u_j = \frac{1}{2d}\sum_i \log 2\pi e \sigma^2_{j,i}$ 估计各 token 不确定性，映射为可靠性权重后自适应加权融合，不依赖额外的不确定性估计器。

## 实验

在 MOSI、MOSEI、CH-SIMS 三个 MSA 基准上评估，文本用预训练 BERT（768 维投影到 128），音频/视频用单层 Transformer 编码器，RTX 4090 单卡训练。论文报告 MIDAS 在多数指标上优于对比方法：

- **MOSI**：Acc-2 71.88/71.26，F1 71.91/71.20，MAE 1.074、Corr 0.534；Acc-2 超 TFR-Net 0.72%，F1 超 EMT-DLFR 1.04%（负/正设置）。MAE 略高于多个基线的最低值 1.065。
- **MOSEI**：全部指标最优，MAE 0.653、Corr 0.607；Acc-5/Acc-7 较 EMT-DLFR 提升 0.77%/0.52%。
- **CH-SIMS**（中文多语言含噪数据）：全部指标最优，Acc-2 73.68%、F1 72.15%，超 EMT-DLFR 2.27%/1.06%，MAE 0.501、Corr 0.427。
- **消融**（表 IV）：去掉 VM/MIM/UAF 中任一组件普遍造成退化，其中 MIM 作用最显著，单独加 UAF 提升有限（表 V：去掉辅助预测损失 $L_{pred}$ 退化最大，MOSI Acc-2 从 71.88 降至 61.96）。
- 图 4 的缺失率曲线显示 MIDAS 随缺失率升高性能退化更平滑。

## 论文自述与课程评估

- **论文自述**：论文将其方法定位为"建立不完全条件下的新 SOTA"；认为 MI 驱动的解耦比 MISA 等基于正交性/对抗的约束式解耦在不完全条件下更有效；实验在随机缺失率设置下进行。
- **本课程评估**：仅在三个 MSA 基准上验证（均属情感分析，非时序预测）；论文未来工作部分自述现实缺失模式更复杂（不对称、时间演化），且未建立异质缺失场景的标准化评估协议；论文提出将 MIDAS 扩展到大规模预训练与更广的多模态任务（情绪识别、人机交互）作为未来方向。

## 关键术语

- **MI minimax**：最小化共享/独有互信息 + 最大化跨模态共享互信息的双向信息论目标
- **interaction information**：三变量互信息，用于把 $I(Z^s; Z^e)$ 拆解为可优化的变分项
- **JSD 估计器**：Deep InfoMax 的互信息变分下界，配合硬负样本
- **UAF（Uncertainty-Aware Fusion）**：以后验方差微分熵为可靠性指标的融合机制

---
