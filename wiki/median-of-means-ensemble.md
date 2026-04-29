---
title: "Median-of-Means Ensemble"
type: technique
tags:
  - ensemble-methods
  - diffusion-models
  - point-estimation
  - robust-statistics
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Median-of-Means (MoM) Ensemble

Median-of-Means (MoM) 估计器是 SimDiff 提出的将扩散模型的概率样本聚合为精确点预测的核心技术 [^src-simdiff]。

## 背景

扩散模型天然探索广泛的可能概率轨迹，其中极端值（outliers）往往不可避免 [^src-simdiff]。如何从这些多样化的概率样本中得出稳定且精确的最终估计，同时仍能忠实捕获整体分布趋势，是一个关键挑战 [^src-simdiff]。

## 方法

MoM 估计器的工作流程：

1. **分割**：将 n 个概率样本分为 K 个子样本，每个子样本大小为 B（通常 B = n/K）
2. **计算子样本均值**：对每个子样本计算均值 μ̂_1, μ̂_2, ..., μ̂_K
3. **取中位数**：计算这 K 个均值的 median(μ̂_1, ..., μ̂_K)
4. **重复**：用随机打乱���数据重复上述过程 R 次
5. **聚合**：对 R 个中位数取平均作为最终估计

数学表达：
$$\hat{\mu}_{MoM} = \frac{1}{R} \sum_{r=1}^{R} \text{median}(\hat{\mu}_1^{(r)}, ..., \hat{\mu}_K^{(r)})$$

## 优势

### 对比简单平均
- 简单平均会平滑掉高频细节，产生过于平滑的轨迹 [^src-simdiff]
- MoM 有效捕获真实数据分布，保留微妙的时间模式 [^src-simdiff]

### 对比单次推理
- 单次推理的 MSE 最高，样本方差大 [^src-simdiff]
- MoM 显著降低 MSE，在所有数据集上取得最低 [^src-simdiff]

### 鲁棒性
- 对离群值和重尾噪声具有显著的鲁棒性 [^src-simdiff]
- 理论上有更强的统计保证，在有限样本情况下提供更紧的收敛界 [^src-simdiff]

## 消融实验结果

| 集成方法 | ETTh1 | Weather | Wind | Caiso |
|---------|-------|---------|------|-------|
| MoM | 0.394 | 0.299 | 0.880 | 0.106 |
| 简单平均 | 0.398 | 0.305 | 0.887 | 0.109 |
| 单次推理 | 0.408 | 0.317 | 0.901 | 0.110 |

MoM 在所有数据集上均取得最低 MSE [^src-simdiff]。

## 理论保证

MoM 估计器源自统计学方法，SimDiff 将其重新引入模型作为从多个概率样本估计真值的可靠方法 [^src-simdiff]。理论上，MoM 提供更强的统计保证，在有限样本情况下提供更紧的收敛界 [^src-simdiff]。

## 与扩散模型的协同

将预训练-条件扩散模型中使用的单次确定性路径替换为 MoM 集成，SimDiff 充分利用了扩散的完整概率轨迹，同时保持数值稳定性 [^src-simdiff]。

[^src-simdiff]: [[source-simdiff]]