---
title: "MIDAS"
type: entity
tags:
  - multimodal-sentiment-analysis
  - incomplete-multimodal-learning
  - mutual-information
  - disentanglement
  - uncertainty-aware-fusion
  - tpami-2026
created: 2026-08-05
last_updated: 2026-08-05
source_count: 1
confidence: medium
status: active
---

# MIDAS

**MIDAS（Mutual Information Disentanglement with uncertainty-Aware fuSion）** 是论文提出的不完全多模态情感分析统一框架（IEEE TPAMI 2026），通过变分建模把每个模态分解为共享与独有因子，用互信息 minimax 目标实现解耦与跨模态对齐，再以不确定性感知融合自适应加权（[^src-midas]）。论文将不完全 MSA 定位为域泛化问题——不同缺失模式诱导不同的输入分布——并主张解耦表示是重构被损坏多模态信息的机制性手段，而非数据填充或启发式协调约束（[^src-midas]）。

## 问题与设计动机

现实多模态输入常因传感器故障、异步、带宽限制或隐私约束而缺失或损坏，导致传统 MSA 模型的表示不稳定、预测退化（[^src-midas]）。论文归纳现有两类处理范式并指出其缺陷（[^src-midas]）：

1. **数据填充类**（TFR-Net、NIAT、CIF-MMIN、EMT）：先重构缺失模态再做下游融合，计算开销大、重构误差累积。
2. **协调表示类**（ALMT、LNLN、P-RMF）：学习跨模态组合下语义一致的表示，但依赖复杂协调机制，仍难以提取任务相关信息。

论文进一步指出两个被忽略的性质（[^src-midas]）：其一，不完全输入更嘈杂、纠缠、语义混杂，正交性约束或对抗目标这类解耦手段只是间接的几何分离，容易不稳定或退化；其二，模态不完整时其表示可靠性下降，而多数融合方法默认所有模态与特征置信度一致，导致不可靠模态主导决策。MIDAS 用互信息直接量化潜在因子间的统计依赖，并以变分后验方差作为可靠性指标分别应对这两个问题。

## 机制

### 1. 变分建模（Variational Modeling, VM）

每个模态 $X_m$（m ∈ {t, a, v}）由共享编码器与独有编码器输出一对多元高斯潜变量（[^src-midas]）：

$$p(Z^s_m | X_m) \sim \mathcal{N}(\mu^s_m, \sigma^s_m I), \quad p(Z^e_m | X_m) \sim \mathcal{N}(\mu^e_m, \sigma^e_m I)$$

用重参数化技巧采样保证可微；方差 $\sigma$ 量化该模态的 aleatoric 不确定性，均值 $\mu$ 为稳定表示，推理时直接用均值。模型预测 log-variance $\log \sigma^2$ 以保数值稳定（[^src-midas]）。

### 2. 互信息 minimax（MI Minimax, MIM）

**最小化解耦项**：目标是最小化 $I(Z^s_m; Z^e_m)$。直接计算不可行，论文利用 interaction information 分解（[^src-midas]）：

$$I(Z^s_m; Z^e_m) = I(Z^s_m; X_m) + I(Z^e_m; X_m) - I(X_m; Z^s_m, Z^e_m)$$

在条件独立假设（给定 $X_m$ 后 $Z^s_m \perp Z^e_m$）下第三项为零，随后用变分编码器与解码器推导出由 KL 散度与重构似然构成的**变分上界**，最小化该上界即实现解耦（[^src-midas]）。

**最大化对齐项**：目标是最大化跨模态共享空间互信息 $I(Z^s_{m_1}; Z^s_{m_2})$。用 Deep InfoMax 的 **JSD 估计器**（softplus 判别器）作为变分下界，比 InfoNCE 更稳健、对负样本数量不敏感；并用独有特征 $Z^e_{m_2}$ 替换负样本构造**硬负样本**，迫使判别器区分共享语义与模态独有信息（[^src-midas]）。

**辅助目标**：单独的 MI 目标不保证共享特征对情感任务语义有效，加入（a）逐模态共享表示的二元分类弱监督损失 $L_{aux}$（用情感标签的粗分类监督）与（b）共享/独有拼接后重构原输入的 MSE 损失 $L_{rec}$，防止共享空间塌缩为平凡共性或噪声模式（[^src-midas]）。

### 3. 不确定性感知融合（Uncertainty-Aware Fusion, UAF）

三个模态的共享/独有特征堆叠为 6 token 序列 $Z = [Z^s_t, Z^s_a, Z^s_v, Z^e_t, Z^e_a, Z^e_v] \in \mathbb{R}^{6 \times d}$（[^src-midas]）。每个 token 的不确定性用对角高斯后验方差的微分熵估计：

$$u_j = \frac{1}{2d}\sum_{i=1}^{d} \log 2\pi e \sigma^2_{j,i}$$

该熵度量尺度不变，跨编码器稳健；$u_j$ 映射为可靠性权重后自适应加权融合，无需额外的不确定性估计器（[^src-midas]）。

## 实验证据

三个 MSA 基准（MOSI、MOSEI、CH-SIMS），文本用预训练 BERT（768 维投影到 128），音频/视频用单层 Transformer 编码器，RTX 4090 单卡，Adam + warm-up + cosine annealing（[^src-midas]）。论文报告（[^src-midas]）：

- **MOSI**：Acc-2 71.88/71.26、F1 71.91/71.20、MAE 1.074、Corr 0.534；Acc-2 超 TFR-Net 0.72%，F1 超 EMT-DLFR 1.04%（负/正设置）。MAE 略高于基线最低值 1.065，但 Corr 显著更高。
- **MOSEI**：全部指标最优，MAE 0.653、Corr 0.607，Acc-5/Acc-7 较 EMT-DLFR 提升 0.77%/0.52%。
- **CH-SIMS**（中文含噪数据）：全部指标最优，Acc-2 73.68%、F1 72.15%，超 EMT-DLFR 2.27%/1.06%，MAE 0.501、Corr 0.427。
- **消融**（表 IV）：MIM 是关键组件；仅 VM+UAF 提升有限，说明有效融合无法补偿表示解耦的缺失。表 V 显示去掉 $L_{pred}$ 退化最大（MOSI Acc-2 71.88→61.96），$L_{dis}$ 与 $L_{ali}$ 次之。
- 图 4 缺失率曲线（r 从 0 到 0.8）显示 MIDAS 退化更平滑稳定。

## 与其他方法的对比

| 维度 | MIDAS | MISA | MMIM | P-RMF | EMT-DLFR |
|------|-------|------|------|-------|----------|
| 解耦手段 | 互信息 minimax（变分上界） | 正交性约束/对抗 | 层次互信息最大化 | 代理驱动融合 | 双水平恢复 |
| 不确定性 | 变分后验方差（微分熵） | 无 | 无 | 无 | 无 |
| 面向场景 | 不完全 MSA | 完整 MSA | 完整 MSA | 不完全 MSA | 不完全 MSA |

论文主张其 MI 驱动的解耦优于 MISA 这类基于损失的分解方法，且 UAF 把后验不确定性显式接入融合，是相对现有不完全 MSA 方法的差别所在（[^src-midas]）。论文自述以上结论基于随机缺失率实验设置。

## 相关页面

- [[source-midas]] — 源文件摘要
- [[mutual-information-disentanglement]] — MI minimax 解耦技术页
- [[mutual-information]] — 互信息概念
- [[contrastive-learning]] — JSD 估计器与 InfoNCE 的对比学习家族
- [[information-bottleneck-principle]] — 信息瓶颈中的互信息压缩
- [[multimodal-semantic-understanding]] — 多模态语义理解与融合策略

[^src-midas]: [[source-midas]]
