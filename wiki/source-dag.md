---
title: "DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - exogenous-variables
  - correlation
  - attention
  - time-series
  - 2026
  - ijcai-2026
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables

**Authors**: Xiangfei Qiu, Yuhan Zhu, Zhengyu Li, Xingjian Wu, Bin Yang, Jilin Hu (East China Normal University, Decision Intelligence Group)

**Venue**: IJCAI 2026 | **Code**: [github.com/decisionintelligence/DAG](https://github.com/decisionintelligence/DAG)

## 核心贡献

DAG 针对 **TSF-X**（Time Series Forecasting with eXogenous variables）提出**双相关网络**：沿时间维和通道维分别**发现**并**注入**外生变量与内生变量之间的相关性结构，端到端确定性预测。[^src-dag]

同一 ECNU 组产出 [[source-timexer|TimeXer]] / DUET / TFB；DAG 延续其外生预测谱系，但切入点是"相关性迁移"而非"patch 融合"或"聚类"。[^src-dag]

## 关键洞察：双相关结构

Figure 2 揭示两类相关性可跨阶段迁移：[^src-dag]

1. **时间相关**：历史外生 → 未来外生的影响结构，**相似于**历史内生 → 未来内生的演化（Granger 因果角度）。
2. **通道相关**：历史外生 ↔ 历史内生的交互模式，**可迁移到**未来外生 ↔ 未来内生（Pearson 相关角度）。

现有方法要么忽略未来外生（[[source-timexer|TimeXer]]、[[source-select-then-balance|CrossLinear]]），要么简单拼接未来外生但无相关性建模（TiDE、TFT），易陷入伪相关。[^src-dag]

## 架构：发现—注入双模块

DAG 由四条网络构成两个对称模块：[^src-dag]

### 时间相关模块（F_θ1 + G_θ2）

- **发现**（F_θ1）：patch-wise Transformer 学历史外生 → 预测未来外生；提取注意力参数 Wq', Wk' 作为时间相关表征。损失 L_t = ‖Y_exo - Ŷ_exo‖。
- **注入**（G_θ2）：**Correlation Trmblock** 在历史内生 → 预测未来内生时，同时用原始 Wq,Wk 与注入的 Wq',Wk' 计算两组注意力分数，用 **gating α** 融合：S_fused = α·σ(QK^T/√d) + (1-α)·σ(Q'K'^T/√d)。

### 通道相关模块（F_θ3 + G_θ4）

- **发现**（F_θ3）：series-wise Transformer 学历史外生 → 预测历史内生；提取 Wq', Wk' 作为通道相关表征。损失 L_c = ‖X_endo - X̂_endo‖。
- **注入**（G_θ4）：Correlation Trmblock 在未来外生 → 预测未来内生时注入 Wq', Wk'。

### Gating 机制

\[
\alpha = \mathrm{MLP}(X^{\text{exo}})^\top \cdot \mathrm{MLP}(X^{\text{endo}})
\]

逐样本自适应权衡原始注意力与注入相关注意力的比重。[^src-dag]

### 融合与训练

最终预测：Ŷ_endo = λ1·Ỹ_endo(时间分支) + (1-λ1)·Ẏ_endo(通道分支)。总损失 L_total = L_f + λ2·(L_t + L_c)，端到端训练。[^src-dag]

## 实验结果

- **12 个 TSF-X 数据集**（5 EPF + 7 自采含水库/风电/能源）：MSE 10/12 第一，MAE 11/12 第一。[^src-dag]
- **无未来外生场景**：用 F_θ1 预测的 Ŷ_exo 替代真实 Y_exo，仍优于 TimeXer、CrossLinear 等仅历史外生方法。[^src-dag]
- **消融**：单用 G_θ2 或 G_θ4 不如联合；加入各自相关模块后均提升；双相关齐用最优。[^src-dag]
- **参数敏感度**：λ1, λ2 ∈ [0.3, 0.7] 最稳；patch 长度 8–32 最优。[^src-dag]
- **增大 lookback**：DAG 随 lookback 增大持续改善，优于基线。[^src-dag]

## 与 KITE 的对比

| | DAG | [[kite|KITE]] |
|--|-----|------|
| 输出 | 确定性点预测 | 概率分布 |
| 骨干 | Transformer | [[flow-matching\|Flow Matching]] |
| 外生利用 | 学习相关性并注入注意力 | 统计先验双线性调制注意力（[[knowledge-guided-conditioning\|KGC]]） |
| 相关性来源 | **可学习** Wq', Wk' 从发现模块提取 | **统计先验** Pearson/Granger 注入 |
| 源分布 | 无 | [[history-conditional-manifold\|HCM]] 可学历史条件源 |
| 条件控制 | gating α 逐样本自适应 | [[classifier-free-guidance\|CFG]] 有/无条件外推 |
| venue | IJCAI 2026 | ICML 2026 |

DAG 与 KITE 在"用相关性指导外生建模"上同向，但 DAG 的相关性是**网络内部跨模块迁移**（discovery → injection），KITE 的相关性是**外部统计知识注入**（prior → attention bilinear）。[^src-dag]

## 局限

- 确定性预测，无不确定性量化。[^src-dag]
- 相关性发现依赖 discovery 模块学到有用 Wq', Wk'；若历史–未来外生结构差异大，迁移效果受限。[^src-dag]
- 无未来外生场景用 F_θ1 预测的 Ŷ_exo 替代，引入额外误差。[^src-dag]

## 相关链接

- [[dag]] — 方法实体
- [[dual-correlation-injection]] — 相关性发现—注入技术
- [[kite]] / [[source-kite]] — 概率外生预测对照
- [[source-timexer]] / [[source-exotst]] / [[source-exost]] / [[source-select-then-balance]] — 外生确定性预测谱系
- [[cross-attention-conditioning]] — 交叉注意条件化
- [[covariate-fusion-module]] — 协变量融合

[^src-dag]: [[source-dag]]
