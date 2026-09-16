---
title: "Inter-Client Dependency"
type: concept
tags:
  - federated-learning
  - traffic-forecasting
  - graph-neural-network
  - data-isolation
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Inter-Client Dependency（跨客户端依赖）

[[federated-traffic-prediction|联邦交通预测]] 中客户端按区域划分，跨区域节点既不在本地邻接矩阵里、也不在本地数据里，这些被隔离邻居的相关性在任一客户端上都不完整。论文把这一缺失称为 missing inter-client dependency，恢复它等价于决定「用什么代表拿不到的节点」[^src-fedhint]。

## 恢复路线

| 路线 | 代表方法 | 代表缺失节点的方式 | 代价 |
|---|---|---|---|
| 服务器预定义图 | CNFGNN | 服务器上的全局图结构 | 需预定义图，需上传额外信息 |
| 服务器聚合中间特征 | FedGTP | 服务器聚合的中间特征 | 上传中间特征，额外通信与隐私风险 |
| 客户端估计缺失节点 | FedGCN | 推断出的缺失节点信息 | non-IID 下推断不准确 |
| 客户端生成代理节点 | [[proxy-node-generation\|FedHINT]] | 本地数据中抽出的 [[hidden-global-components\|隐藏全局成分]] | 只上传模型参数，参数量与训练时间增加 |
| 不恢复 | FedGRU、CTFL、FedTPS | 无 | 跨区域相关性完全缺失 |

五条路线按论文 Introduction 与 Related Work 的叙述归并[^src-fedhint]。

## 检验方式

论文把「本地数据生成的代理节点」与「全局数据生成的代理节点」做余弦相似度，PEMS03 上矩阵对角占优，说明本地抽出的成分与全局数据导出的版本接近[^src-fedhint]。论文没有报告恢复程度的直接度量（例如与真实跨区域节点相关性的对照）[^src-fedhint]。

## 相关页面

- [[federated-traffic-prediction|Federated Traffic Prediction]]
- [[hidden-global-components|Hidden Global Components]]
- [[proxy-node-generation|Proxy Node Generation (GPN)]]
- [[source-fedhint|FedHINT]]

[^src-fedhint]: [[source-fedhint]]
