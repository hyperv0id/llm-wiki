---
title: "Frequency Balance Loss"
type: technique
tags:
  - regularization
  - routing
  - load-balancing
  - frequency-domain
  - loss-function
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Frequency Balance Loss

**Frequency balance loss** 是 [[adafre|AdaFre]] 在预测损失之外附加的负载均衡正则：统计各频率分支在一个 mini-batch 上被选中的平均概率，惩罚其偏离均匀分布，以防止路由器坍缩到少数几个频率[^src-adafre]。

## 动机

论文给出的失效模式是：没有额外约束时，频率路由器可能过拟合到少数主导频率（例如总是选择低频趋势），即便其他频率在少数但重要的场景中同样有用[^src-adafre]。

## 定义

设 $\hat{r}_{jp}^{(i)}$ 为第 $i$ 个样本中第 $j$ 个节点对第 $p$ 个频率的选择概率，$B$ 为 batch size，则该频率在 batch 上的平均选择概率为[^src-adafre]：

$$\tilde{r}_p=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{N}\hat{r}_{jp}^{(i)}$$

均衡损失惩罚它偏离 $1/P$[^src-adafre]：

$$L_{\text{bal}}=\sum_{p=1}^{P}\left(\tilde{r}_p-\frac{1}{P}\right)^2$$

总目标为 $L_{\text{total}}=L_{\text{pred}}+L_{\text{bal}}$，两项之间没有报告权重系数，即隐式取等权[^src-adafre]。

式(14) 对 batch 与节点双重求和（$\sum_{i=1}^{B}\sum_{j=1}^{N}$），但归一化因子只除以 $B$ 而非 $B\cdot N$，量纲上与"平均概率"不符——见 [[source-adafre]] 的不一致记录[^src-adafre]。

## 与 MoE 负载均衡的关系

这与稀疏混合专家中防止专家坍缩的负载均衡项属同一族思路，但作用对象是频率分支而非专家网络：

| 方法 | 平衡对象 | 平衡信号的构造 |
|------|----------|----------------|
| **Frequency balance loss**（AdaFre） | $P$ 个时间频率分支 | 逐频率的 batch 平均选择概率，目标为均匀分布 $1/P$ |
| [[sparse-balanced-mixture-of-experts-st\|MAGE 稀疏专家]] | 图学习专家 | 每节点 top-$K$ 激活 + 符号 SGD 负载均衡 |
| [[ams-moe\|AMS-MoE]] | 时间尺度 expert | 噪声注入 top-$K$ 路由，论文未采用显式平衡损失 |

## 使用边界

- 论文报告三项消融（去分带、共享谱嵌入、去自适应路径），但没有单独去掉 $L_{\text{bal}}$ 的消融，因此均衡项本身的贡献缺少直接消融证据[^src-adafre]。
- 论文未报告训练后的频率选择分布，也没有给出"无均衡项时路由是否真的坍缩"的对照[^src-adafre]。
- 均衡目标是全频带均匀，而非按频带有用性加权；论文未讨论这种均匀假设在频率重要性高度偏斜的数据上的适配性[^src-adafre]。

## 相关页面

- [[adafre]] — 使用该损失的模型
- [[frequency-pathway-routing]] — 被正则的路由机制
- [[sparse-balanced-mixture-of-experts-st]] — 稀疏平衡专家
- [[mixture-of-experts]] — MoE 架构与路由概念
- [[source-adafre]] — 源文件摘要

[^src-adafre]: [[source-adafre]]
