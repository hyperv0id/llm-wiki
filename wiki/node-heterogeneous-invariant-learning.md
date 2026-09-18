---
title: "Node-Heterogeneous Invariant Learning"
type: technique
tags:
  - traffic-forecasting
  - invariant-learning
  - conditional-computation
  - contrastive-learning
  - spatio-temporal
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: low
status: active
---

# Node-Heterogeneous Invariant Learning

**节点异质不变学习**是 [[dynastar|DynaSTar]] 的预测端机制：常规模型用全局共享的预测映射，学到的是压制节点特异信号的平均模式；该机制把预测器拆成共享与节点特异两部分，在逐节点个性化的同时保持跨环境不变[^src-dynastar]。

## 机制

### 邻域环境集构造

对节点 $i$，其 $t$ 时刻邻域由多元 Bernoulli 分布参数化（各方向边独立）：
$$P(\mathcal{N}^t_i)=\prod_{j\neq i}(\hat{A}^t_{ij})^{a^t_{ij}}(1-\hat{A}^t_{ij})^{1-a^t_{ij}}\cdot\prod_{j\neq i}(\hat{A}^t_{ji})^{a^t_{ji}}(1-\hat{A}^t_{ji})^{1-a^t_{ji}}$$
每个训练 batch 从中经 Straight-Through Gumbel-Softmax 采样 $p_e{=}2$ 张稀疏图，构成环境集 $\mathbb{E}$。与全局随机扰动不同，环境直接取自动态概率图，逐节点模拟异质拓扑漂移，迫使模型提取跨邻域上下文的节点级不变预测模式[^src-dynastar]。

### 节点特异调制

条件向量 $c^t_i=\mathrm{MLP}(z^t_i+e_{time}(t))$（64 维，hidden 8，$z^t_i$ 为动态表示）作为节点的自适应编码；一个调制生成 MLP 把 $c^t_i$ 变成预测器逐层仿射参数 $w^{(l)}_i,b^{(l)}_i$，对与编码器同构的共享预测器中间表示做 Feature-wise Linear Modulation（FiLM，Perez et al. 2018）式调制：$\tilde{I}^{(l)}_i=w^{(l)}_i\odot I^{(l)}+b^{(l)}_i$；各层调制输出投影拼接后投影出节点 $i$ 的预测。这样共享组件承载全局可迁移模式，仿射参数承载节点个性化知识[^src-dynastar]。

### 优化目标

- **节点异质性对比损失** $L_{node}$（InfoNCE，温度 1）：batch 内同节点跨时刻的条件向量为正对、跨节点为负对，蒸馏稳定的节点专属向量（权重 $\gamma_1{=}0.01$）[^src-dynastar]。
- **节点级不变损失** $L_{inv}$：对节点 $i$ 在各采样环境下的预测风险 $R^e_i(\Phi)$ 施加方差惩罚（IRM 风格，引 Arjovsky et al. 2019；权重 $\gamma_2{=}1$），驱动模型在连接性波动下提取稳定的节点级预测模式[^src-dynastar]。
- 总目标 $L_{total}=L_{pred}+0.01\,L_{node}+1\,L_{inv}$，$L_{pred}$ 为全部采样环境上的平均预测损失[^src-dynastar]。

## 证据与边界

消融中去掉节点调制（w/o NM，SD 长期 60min MAE 26.87→28.25）与去掉邻域环境构造（w/o NE，→27.96）均掉点，论文以此论证节点级异质建模与邻域不变表示的必要性[^src-dynastar]；在未见图零样本（NWGBA/NEGBA）上优于 STONE/STEVE，论文归因于该机制在训练中捕获了"特异且可迁移"的模式[^src-dynastar]。边界：环境数 $p_e$、FiLM 容量的敏感性未报告；与 [[personalized-aggregation-strategy|FedHINT 的个性化聚合]]同属"共享 + 节点特异"分解思路但目标不同（跨域泛化 vs 联邦聚合）。

## 引用

[^src-dynastar]: [[source-dynastar]]
