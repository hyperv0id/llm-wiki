---
title: "Dual Correlation Injection"
type: technique
tags:
  - attention
  - correlation
  - exogenous-variables
  - knowledge-transfer
  - time-series
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# Dual Correlation Injection

**Dual Correlation Injection** 是 [[dag|DAG]] 的核心技术：沿时间维和通道维分别**发现**外生–内生相关性（提取注意力 Wq', Wk'），再**注入**到内生预测的注意力计算中，用 gating α 自适应融合原始与注入注意力。[^src-dag]

## 动机

外生预测中存在可迁移的相关性结构（Figure 2）：[^src-dag]

- **时间维**（Granger 因果）：历史外生 → 未来外生 ≈ 历史内生 → 未来内生。
- **通道维**（Pearson 相关）：历史外生 ↔ 历史内生 ≈ 未来外生 ↔ 未来内生。

但朴素地把统计权重加到输入端不解决结构瓶颈——注意力投影空间不随相关环境移动。DAG 改为跨模块**迁移注意力参数本身**。[^src-dag]

## 发现：提取 Wq', Wk'

### 时间相关发现（F_θ1）

patch-wise Transformer 学历史外生 → 未来外生：[^src-dag]

\[
Q = S_i W_{q'},\ K = S_i W_{k'},\ V = S_i W_{w'}
\]

训练后提取 Wq', Wk' 作为时间相关表征；预测损失 L_t = ‖Y_exo - Ŷ_exo‖。[^src-dag]

### 通道相关发现（F_θ3）

series-wise Transformer 学历史外生 → 历史内生，同样提取 Wq', Wk'；损失 L_c = ‖X_endo - X̂_endo‖。[^src-dag]

## 注入：Correlation Trmblock

在内生预测的 Transformer block 中，同时计算原始注意力与注入注意力：[^src-dag]

\[
Q = P_i W_q,\ K = P_i W_k,\ V = P_i W_v \quad(\text{原始})
\]
\[
Q' = P_i W_{q'},\ K' = P_i W_{k'} \quad(\text{注入，来自发现模块})
\]

融合注意力分数：[^src-dag]

\[
S_{\text{fused}} = \alpha\cdot\sigma\!\left(\frac{QK^\top}{\sqrt{d}}\right) + (1-\alpha)\cdot\sigma\!\left(\frac{Q'K'^\top}{\sqrt{d}}\right)
\]

\(\alpha\) 越大越依赖原始（数据驱动）注意力；\(\alpha\) 越小越依赖注入（相关性驱动）注意力。[^src-dag]

## Gating：逐样本自适应

\[
\alpha = \mathrm{MLP}(X^{\text{exo}})^\top \cdot \mathrm{MLP}(X^{\text{endo}})
\]

两个 MLP 分别编码外生与内生，点积输出标量 α。逐样本动态决定相关性注入强度——相关性强的样本 α 小（注入多），弱的样本 α 大（原始多）。[^src-dag]

时间模块与通道模块的 gating 输入对略有不同：[^src-dag]

- 时间注入 gating：MLP(历史外生) · MLP(历史内生)
- 通道注入 gating：MLP(历史外生) · MLP(未来外生)

## 与 KITE KGC 的对比

| | DAG Correlation Injection | [[knowledge-guided-conditioning\|KITE KGC]] |
|--|---------------------------|------|
| 相关性来源 | 网络内部跨模块学习 Wq',Wk' | 外部统计先验 Pearson/Granger |
| 注入位置 | 注意力 Q,K 投影 | 注意力 q(W1+s·W2)k 双线性 |
| 融合方式 | gating α 融合两组 softmax 分数 | 先验门控 + log-gating 调制单一 softmax |
| 自适应 | 逐样本 α | 逐元素 s_ij |
| 预测范式 | 确定性 | 概率（+ [[classifier-free-guidance\|CFG]]） |

DAG 的相关性是**网络内部迁移**（discovery→injection）；KITE 的相关性是**外部知识注入**（statistical prior→attention bilinear）。两者在"把相关性写进注意力投影"上同构，但知识来源与融合点不同。[^src-dag]

## 设计要点

1. **发现模块有双重用途**：既产生相关损失（L_t / L_c），又提取 Wq', Wk' 供注入。[^src-dag]
2. **注入不改 V**：只替换 Q, K 投影，V 仍用原始 Wv——保留值空间的数据驱动性。[^src-dag]
3. **双分支融合**：最终预测 λ1·Ỹ(时间) + (1-λ1)·Ẏ(通道)，两路互补。[^src-dag]
4. **无未来外生退化**：用 F_θ1 预测的 Ŷ_exo 替代真实 Y_exo 喂 G_θ4，保持框架完整。[^src-dag]

## 相关页面

- [[dag]] / [[source-dag]]
- [[knowledge-guided-conditioning]] — KITE 的统计先验注入
- [[cross-attention-conditioning]] — 一般交叉注意条件化
- [[covariate-fusion-module]] — 协变量融合
- [[kite-manifold-guidance-chain]] — KITE 三件套串联
- [[spurious-patterns]] — 伪相关问题

[^src-dag]: [[source-dag]]
