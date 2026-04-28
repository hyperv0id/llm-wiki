---
title: "Attention Dilution"
type: technique
tags:
  - attention
  - long-context
  - theoretical-analysis
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Attention Dilution

**Attention Dilution**（注意力稀释）是 Gu 等人 (2026) 提出的理论机制，用于解释大语言模型在长上下文场景下性能退化的根本原因[^src-paperbench]。

## 核心思想

Soft attention 对所有 token 进行归一化聚合，导致当上下文长度增加而任务相关 token 数量保持固定时，相关 token 必须与越来越多的无关 token 竞争有限的注意力预算[^src-paperbench]。

## 数学形式

### 注意力聚合

给定查询 $q$ 和上下文 $C_n = \{x_1, ..., x_n\}$，self-attention 计算压缩表示[^src-paperbench]：

$$h(q, C_n) = \sum_{i=1}^{n} \alpha_i v_i$$

其中注意力权重：

$$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}, \quad s_i = q^\top k_i$$

### 定理 6.1 (Attention Dilution under Context Scaling)

设上下文 $C_n$ 包含任务相关子集 $R \subset [n]$，$|R| = m$（$m$ 固定，独立于 $n$），其余为无关 token $N = [n] \setminus R$[^src-paperbench]。

假设：
1. $\{s_i\}_{i \in R}$ i.i.d. 来自 $D_r$，$\{s_i\}_{i \in N}$ i.i.d. 来自 $D_n$，独立[^src-paperbench]
2. 有限指数矩：$\mu_r = \mathbb{E}[\exp(S_r)] < \infty$，$\mu_n = \mathbb{E}[\exp(S_n)] < \infty$，$\mu_n > 0$[^src-paperbench]

定义分配给相关 token 的总注意力质量：

$$\text{AR}(n) = \sum_{i \in R} \alpha_i$$

则当 $n \to \infty$ 时[^src-paperbench]：

$$\text{AR}(n) = O_p\left(\frac{1}{n}\right)$$

**证明思路**：令 $Z_i = \exp(s_i)$，则 $\text{AR}(n) = \frac{\sum_{i \in R} Z_i}{\sum_{i \in R} Z_i + \sum_{i \in N} Z_i}$。由大数定律，$\sum_{i \in N} Z_i = \Theta_p(n)$，而 $\sum_{i \in R} Z_i = O_p(1)$（$|R| = m$ 固定），故结论成立[^src-paperbench]。

## 推论：统一的长上下文性能退化

设目标 $Y$ 满足 $Y = g(\{x_i: i \in R\})$，模型预测 $\hat{Y} = f(q, h(q, C_n))$，其中 $f$ 对第二个参数 Lipschitz 连续[^src-paperbench]。

则对于任意仅在 $R$ 上不同的两个上下文 $C_n$ 和 $C_n'$[^src-paperbench]：

$$f(q, h(q, C_n)) - f(q, h(q, C_n')) \xrightarrow{p} 0$$

**含义**：当上下文增长时，模型对任务相关内容的微小变化越来越不敏感[^src-paperbench]。

## 关键洞察

### 1. 固定注意力预算

Attention 执行的是 soft normalized aggregation，而非 hard selection。任务相关 token 和无关 token 直接竞争固定的注意力预算[^src-paperbench]。

### 2. 稀疏信息表征

个性化依赖用户偏好/约束 $R$（稀疏），隐私推理依赖敏感信息 $R$（稀疏）。当 $|R| \ll n$ 时，$R$ 的贡献被稀释[^src-paperbench]。

### 3. 信息论解释

$h(q, C_n)$ 与目标 $Y$ 之间的互信息 $I(Y; h(q, C_n))$ 随 $n$ 增长而降低，限制了模型基于任务相关信息进行预测的能力[^src-paperbench]。

## 特殊情况：因果掩码

**Remark 6.2**：在 decoder-only Transformer 中，如果所有无关 token 连续出现在序列尾部，即 $R \subseteq \{1, ..., m\}$，$N = \{m+1, ..., n\}$，且查询位置 $t \leq \max(R)$，则因果掩码阻止查询关注 $N$ 中的 token[^src-paperbench]。

此时 $\text{AR}(n) = O_p(1)$，注意力稀释被避免[^src-paperbench]。

**实践意义**：任务相关信息应尽量靠近查询位置（序列尾部）以获得更好的长上下文性能[^src-paperbench]。

## 与其他理论的关系

| 理论 | 解释对象 | 与 Attention Dilution 的关系 |
|------|----------|-------------------------------|
| NTK-aware / YaRN | 位置编码外推 | 互补：解决位置编码问题，Attention Dilution 解决注意力分配问题 |
| ALiBi | 位置编码外推 | 互补：线性偏置改善外推，但不解决注意力稀释 |
| 上下文窗口扩展 | 模型架构 | 互补：扩展上下文长度，但 Attention Dilution 揭示更深层问题 |

## 引用

[^src-paperbench]: [[source-paperbench]]