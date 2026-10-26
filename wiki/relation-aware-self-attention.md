---
title: "Relation-aware Self-Attention"
type: concept
tags:
  - transformer
  - attention
  - graph
  - relative-position
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Relation-aware Self-Attention

Relation-aware self-attention 把自注意力从「元素对元素的内容匹配」推广为「带标签边上的匹配」：输入建模为标号有向全连接图，任意两个元素 $x_i$、$x_j$ 之间有一条边 $a_{ij}\in\mathbb{R}^{d_a}$，注意力在原有 key/value 上加入该边向量[^src-shaw-2018-relative-position]。自注意力本身是其特例（所有边取 0），序列相对位置表示是另一特例（边标签退化为裁剪后的相对偏移）。

## 形式

设 $x_i\in\mathbb{R}^{d_x}$、$W^Q,W^K,W^V\in\mathbb{R}^{d_x\times d_z}$，$d_a$ 为边向量维。两条注入路径：

$$\begin{aligned}
z_i &= \sum_{j}\alpha_{ij}\big(x_j W^V + a^V_{ij}\big) && \text{// } [1, \bcancel{n}]\times[\bcancel{n}, d_z]\to[1, d_z]\ \text{——式 (3)：边向量与 value 走同一条加权通路}\\
e_{ij} &= \frac{1}{\sqrt{d_z}}\,x_i W^Q\big(x_j W^K + a^K_{ij}\big)^{\top} && \text{// } [d_z]\cdot[d_z]\to\text{标量}\ \text{——式 (4)：边向量先加到 key，打分因此依赖关系类型}
\end{aligned}$$

两处都用加法而非新的线性变换，将边信息融入已有表示可避免显著增加计算量[^src-shaw-2018-relative-position]。

两条路径的作用位置不同：$a^K_{ij}$ 进入打分，改变「$i$ 看谁」的权重；$a^V_{ij}$ 进入被加权的量，改变「取回什么」。前者只要边类型不同就改变 logit，后者在输出加权和里留下关系向量。

## 边的取值：通用图与序列

边向量可以每条边独立，也可以由少量标签生成：

$$\begin{aligned}
a^K_{ij} &= \theta^K_{ij}\in\mathbb{R}^{d_a} && \text{// 通用图：每条边一个自由参数，} n^2 \text{ 条边各占一个向量}\\
a^K_{ij} &= w^K_{\operatorname{clip}(j-i,\,k)} && \text{// 序列：} 2k+1 \text{ 个标签，与 } n \text{ 无关（详见 [[relative-position-representation]]）}
\end{aligned}$$

序列是论文唯一实测的情形。裁剪让边向量表大小与序列长度无关，超出 $k$ 的距离共用同一标签[^src-shaw-2018-relative-position]。

## 空间代价

$n^2$ 条边各带一个 $d_a$ 向量时，边表示占用的空间为 $O(hn^2d_a)$；论文按头共享后降至 $O(n^2d_a)$，自注意力整体空间从 $O(bhnd_z)$ 增至 $O(bhnd_z+n^2d_a)$。将边引入注意力模型的空间开销，论文引 Veličković 等（2017）的图注意力为据[^src-shaw-2018-relative-position]。

## 两条路径的消融

base 模型、EN-DE 开发集（newstest2013）上分别关掉两条注入：

| $a^V$ | $a^K$ | BLEU |
|---|---|---|
| 是 | 是 | 25.8 |
| 否 | 是 | 25.8 |
| 是 | 否 | 25.3 |
| 否 | 否 | 12.5 |

仅保留 $a^K$ 即可维持全部增益，反之下降 0.5；两者均关闭时位置信息完全缺失，BLEU 降至 12.5[^src-shaw-2018-relative-position]。论文据此提出，仅在兼容性函数中注入边信息可能已经足够，并把该结论是否适用于其他任务留待验证[^src-shaw-2018-relative-position]。

## 未来工作

论文将该形式作为可推广到任意有向标号图输入的接口，并将非线性兼容性函数（$e_{ij}$ 不再仅为内积）列为后续方向；两者均以高效实现为前提[^src-shaw-2018-relative-position]。

## 与 GPE 框架的关系

边向量落在 key、value 的表示层，位置项 $p_{ij}=q_i(a^K_{ij})^{\top}/\sqrt{d_z}$ 随 query 内容变化且不经 $x_j$ 投影[^src-shaw-2018-relative-position]。在 [[generalized-positional-encoding-framework|GPE]] 的 $f(n)\cdot q_i^{\top}W(n)k_j+b(n)$ 形式里它没有对应项：既不是与 $q_i$ 无关的标量 $b(n)$，也不是作用于 $k_j$ 的 $W(n)$[INFERENCE]。

## 相关页面

[[source-shaw-2018-relative-position]] 概述论文摘要；[[relative-position-representation]] 为序列上的特化，含 $2k+1$ 个裁剪标签；[[generalized-positional-encoding-framework]] 涵盖位置编码的乘性变换与加性偏置分解；[[alibi]] 为只保留加性标量偏置的简化；[[transformer-architecture]] 则对应被推广的原结构。

[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]（arXiv:1803.02155v1，PDF 归档 `raw/shaw-2018-relative-position-1803-02155.pdf`）
