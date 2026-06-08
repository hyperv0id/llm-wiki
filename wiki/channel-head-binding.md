---
title: "Channel-Head Binding (CHead Attention)"
type: technique
tags:
  - time-series
  - data-imputation
  - attention
  - cnn-transformer
  - iclr-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Channel-Head Binding（CHead Attention）

**Channel-Head Binding** 是 [[t1|T1]] (ICLR 2026) 的核心机制：在 CNN 通道与注意力头之间建立**一对一对应**（$n_h = C$），使每个注意力头只处理它绑定的那个通道、跨所有变量做注意力[^src-t1]。这把"特征级选择性"引入跨变量信息传递，是 T1 在重度缺失下鲁棒插补的关键。

## 动机：朴素 CNN+Attention 的不足

T1 用共享 depthwise 卷积为每个变量提取时间特征，再用变量轴注意力做跨变量融合。但若把整条变量当作**单个 token**（如 [[itransformer|iTransformer]] 的做法），其所有通道被迫混合——当缺失只破坏了某些**特定时间特征**时，无法把"被污染的特征"与"可靠的特征"隔离开来再传递[^src-t1]。这需要特征级的细粒度控制。

## 机制

前提：**共享卷积滤波器**。Temporal DWConv 的权重在所有变量间共享，于是同一通道 c 从每个变量提取的是**同一类型**的时间模式——通道之间语义对齐。这使"按通道分别做跨变量注意力"有意义[^src-t1]。

对每个通道 $c \in \{1,...,C\}$，仅在变量轴上做注意力：

$$O_c = \text{Softmax}\!\left(\frac{Q_c K_c^T}{\sqrt{L}}\right) V_c, \qquad Q_c, K_c, V_c \in \mathbb{R}^{M \times L}$$

其中 $M$ 为变量数，$L$ 为潜在时间长度。每个 head（= 通道）独立判断"该类时间特征在各变量间如何传递"。各通道输出 $\{O_1,...,O_C\}$ 沿通道维拼接为 $O \in \mathbb{R}^{M \times C \times L}$，再经 PWConv + LayerNorm + 残差。

## 为何对缺失鲁棒

当缺失使某通道无法观测到其专属模式时，该通道提取的特征**变得不可靠**；与之绑定的注意力头便**自动降低对该通道的依赖**，而特征级的隔离防止这种局部不确定性**污染其他通道**[^src-t1]。论文的表征分析证实：

- 目标变量缺失率↑ → 其他变量对它的注意力权重↓；浅层最敏感（第 1 层 −46%，末层 −6%）[^src-t1]。
- 注意力调节取决于**哪些时间模式仍可观测**——去除高方差区域使注意力降 10.4%，去除低方差区降 7.5%——而非仅由缺失率决定[^src-t1]。

## 消融证据

- **1-to-1 绑定的必要性**：把每个 head 绑定的通道数改为 8/16/32（而非默认 1）分别使 MSE 升高 7.45%/16.86%/14.57%[^src-t1]。
- **自适应 vs 固定传递**：把 CHead Attention 换成 pointwise 卷积 +12.91%；完全去掉跨变量建模 +56.16%——既证明跨变量信息关键，也证明注意力式自适应传递优于卷积式固定传递[^src-t1]。

## 复杂度

每个通道在 $M$ 个变量间做注意力（特征维 $L$），总复杂度 $O(M^2 \cdot C \cdot L)$——与减少头数、按比例增大每头维度的标准多头注意力同阶[^src-t1]。

## 与相关机制的对比

- **vs 标准多头注意力 (MHA)**：MHA 各头维度是 embedding 的切分，头与"语义特征"无显式绑定；CHead 强制 head ↔ CNN 通道（= 一类时间特征）一一对应。
- **vs [[multivariate-correlation-attention|iTransformer 变量注意力]]**：iTransformer 在变量 token 间做注意力但融合了整条序列；CHead 在**每个时间特征通道上分别**做变量注意力，保留特征级选择性。
- **vs [[channel-independence|Channel Independence]]**：CI 完全不建模跨变量关系；CHead 则做**特征级**的跨变量传递。

## 关联页面

- [[t1]] — 提出此机制的插补模型
- [[itransformer]] — 变量轴注意力的前身（T1 在其上增加特征级绑定）
- [[multivariate-correlation-attention]] — iTransformer 的变量相关性注意力
- [[channel-independence]] — 跨变量建模谱系的另一端

[^src-t1]: [[source-t1]]
