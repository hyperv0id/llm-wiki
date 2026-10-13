---
title: "Isotropic Gaussian Embeddings"
type: concept
tags:
  - self-supervised-learning
  - representation-learning
  - statistics
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# 各向同性高斯嵌入分布

自监督编码器的嵌入应服从什么分布？LeJEPA 给出的回答：在协方差受标量约束的分布族中，各向同性高斯是唯一使 worst-case 下游预测风险最小的嵌入分布[^src-lejepa]。这一结果把「嵌入不应退化」从设计直觉变成有具体目标分布的可优化命题，是 [[sigreg|SIGReg]] 的正则目标与 [[lejepa|LeJEPA]] 单超参训练的理论根据。

## 问题

foundation model 训练完成后要面对训练时未知的下游任务，嵌入分布必须在「对所有下游任务」的意义上可用，而不是对某个具体任务最优。LeJEPA 把这一要求形式化为 worst-case 风险，并按探针类型分两条路线推导：线性探针（回归/分类）与非线性探针（k-NN、kernel）[^src-lejepa]。

## 机制

**线性探针：偏差与方差分开看。** 把下游预测写成嵌入上的线性模型：

- 偏差（lemma 1）：估计参数与真值参数的余弦相似度仅在嵌入分布各向同性时等于 1；各向异性时无论样本量多大、正则多强都退化[^src-lejepa]。
- 方差（lemma 2）：协方差各向异性时，学到的参数跨训练集波动更大。二维两类逻辑回归上重复采样训练集拟合，各向同性嵌入的 $\mathrm{Var}(\hat\beta)=0.0056$，各向异性为 $0.0801$，各向异性下学到的决策边界在采样间大幅摆动（Figure 3）[^src-lejepa]。

**非线性探针：积分平方偏差（Theorem 1）。** k-NN 探针的积分平方偏差分解为 $\frac{r_0^4}{(K+2)^2 h^2\mu(K)^2}\tau_g^2 J(p)+O(r_0^4)$，kernel 探针上界为 $\frac{h^4}{2}(2B^2+8L^2J(p))+o(h^4)$，其中 $J(p)$ 是嵌入密度的泛函；在协方差受标量约束的分布族内，各向同性高斯唯一最小化该项[^src-lejepa]。附录把同型结论推广到 k-NN 与 kernel 探针的一般形式（Theorems 7、8）[^src-lejepa]。

两个误差来源指向同一个解：各向同性消除「某些方向信息弱于其他方向」的结构，探针无论取哪个方向读出信息，看到的信噪结构一致；论文在 3.3 节把这一性质定位为「下游任务信息不可得时的最优设计」[^src-lejepa]。

## 适用边界

- 「唯一最优」的表述相对协方差受标量约束的分布族成立，脱离该族约束的分布不在定理覆盖范围内[^src-lejepa]。
- k-NN 与 kernel 定理依赖附录 A 列出的密度正则性条件[^src-lejepa]。
- worst-case 是跨下游任务族的概括；论文自己的限定是无下游任务信息时的设计准则，不声称对特定任务最优[^src-lejepa]。

## Related Pages

- [[lejepa]] — 以该分布为目标的 JEPA 训练
- [[sigreg]] — 该分布约束的可微实现
- [[jepa]] — JEPA 架构背景
- [[non-contrastive-ssl]] — 防塌缩路线谱系

[^src-lejepa]: [[source-lejepa]]
