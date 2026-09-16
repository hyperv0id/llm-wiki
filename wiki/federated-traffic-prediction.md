---
title: "Federated Traffic Prediction"
type: concept
tags:
  - federated-learning
  - traffic-forecasting
  - privacy-preserving
  - distributed-training
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Federated Traffic Prediction（联邦交通预测）

多个客户端各持本地路网的交通数据，只上传模型参数、不上传原始数据[^src-fedhint]。

## 设定

全局路网 $G=\langle V,E\rangle$，$M$ 个客户端各维护局部子图 $G_m=\langle V_m,E_m\rangle$ 与数据 $\mathcal{D}_m=\{x_t\}_{t=1}^T$（$x_t\in\mathbb{R}^{|V_m|}$）。客户端本地训练 $f_{W_m}$ 做 $T_1\to T_2$ 预测，服务器按节点数加权聚合：

$$\arg\min_{W_1,\cdots,W_M}\sum_{m=1}^{M}\frac{|V_m|}{|V|}\mathcal{L}(W_m,\mathcal{D}_m),$$

聚合后回发参数进入下一轮（式 2）[^src-fedhint]。

## 两条约束

- 客户端只看得到本地子图，跨区域传感器相关性在任何客户端上都不完整，即 [[inter-client-dependency|missing inter-client dependency]]。集中式 STGNN 没有这个问题：它的邻接矩阵覆盖全部传感器[^src-fedhint]。
- 各区域交通数据非独立同分布（non-IID），共享并平均全部参数会牺牲区域特有信息[^src-fedhint]。

## 评估协议

论文与全部基线统一采用：METIS 把全局路网切成 6 个子图作为客户端；训练/验证/测试 6:2:2；12 步输入预测 12 步；每个客户端 2 轮本地训练、共 200 轮全局；Adam，学习率 0.003，batch 64；MAE/RMSE 逐客户端评估后跨客户端平均[^src-fedhint]。

## 相关页面

- [[source-fedhint|FedHINT]]
- [[inter-client-dependency|Inter-Client Dependency]]
- [[personalized-aggregation-strategy|Personalized Aggregation Strategy]]
- [[traffic-forecasting|Traffic Forecasting]]

[^src-fedhint]: [[source-fedhint]]
