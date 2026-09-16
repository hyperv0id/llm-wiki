---
title: "Hidden Global Components"
type: concept
tags:
  - federated-learning
  - traffic-forecasting
  - global-information
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Hidden Global Components（隐藏全局成分）

论文假设本地交通数据本身含有反映跨区域交通变化的成分，可以在本地抽出来代表全局信息[^src-fedhint]。

## 机制

跨客户端共享 $N$ 个 global queries $Q\in\mathbb{R}^{N\times d_{att}}$：局部历史 $X_t$ 线性映射出 key/value，经 [[time-shifted-filters|time-shifted filters]] 在频域提取时序特征，由 $Q$ 做 scaled dot-product attention，再线性投影回时域得到 $N$ 个 [[proxy-node-generation|代理节点]]。$N$ 就是全局成分的个数[^src-fedhint]。

成分之间用正交正则拉开：

$$L_{div}=\frac{1}{N(N-1)}\sum_{i=1}^{N}\sum_{j=i+1}^{N}|q_i^\top q_j|,\qquad \mathcal{L}=\sum_t|\hat{x}_t-x_t|+\lambda L_{div},\quad \lambda=0.1.$$

约束只作用在 query 上[^src-fedhint]。

## 证据

| 证据 | 观测 |
|---|---|
| 超参 | PEMS03 上 $N=64$ 最优，论文称最优 $N$ 与传感器数 $|V|$ 正相关[^src-fedhint] |
| 相似度 | 本地生成的代理节点与全局数据生成的代理节点余弦相似度矩阵对角占优[^src-fedhint] |
| 消融 | 移除代理节点后 PEMS03 MAE 11.95→12.70、RMSE 19.12→20.39[^src-fedhint] |

论文未给出这些成分与具体物理量的对应，「反映跨区域交通变化」是定性描述[^src-fedhint]。

## 相关页面

- [[inter-client-dependency|Inter-Client Dependency]]
- [[proxy-node-generation|Proxy Node Generation (GPN)]]
- [[time-shifted-filters|Time-Shifted Filters]]

[^src-fedhint]: [[source-fedhint]]
