---
title: "Personalized Aggregation Strategy (PAS)"
type: technique
tags:
  - federated-learning
  - traffic-forecasting
  - parameter-sharing
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Personalized Aggregation Strategy (PAS)

FedHINT 不聚合全部参数，只聚合承载全局信息的那部分[^src-fedhint]。

## 划分

| 组别 | 模块 | 处理 |
|---|---|---|
| shared $W^s_m$ | 隐藏全局成分提取器、global encoder | 上传服务器聚合 |
| private $W^p_m$ | local encoder、predictor | 留在客户端 |

global 侧负责抽全局成分、恢复跨客户端依赖，共享能增强利用全局信息的能力；private 侧负责本地路网内部相关性与本地预测，留在本地以对抗 non-IID[^src-fedhint]。

## 聚合

$$\bar{W}^s\leftarrow\sum_{m=1}^{M}\frac{|V_m|}{|V|}W^s_m,$$

即按客户端节点数占全局节点数的比例加权，每轮本地训练后执行并回发（式 15）[^src-fedhint]。

## 实证

改为共享全部参数（w/o PAS）后，PEMS03 MAE 11.95→12.52、RMSE 19.12→20.19，四个数据集一致掉点[^src-fedhint]。

## 与 FedTPS 的区别

FedTPS（同一第一作者的前作）在个性化联邦里共享公共交通模式；FedHINT 共享的是负责全局信息与跨客户端依赖的整块模块。FedTPS 在本文的分类里属于「不建模跨客户端依赖」一类[^src-fedhint]。

## 相关页面

- [[federated-traffic-prediction|Federated Traffic Prediction]]
- [[proxy-node-generation|Proxy Node Generation (GPN)]]
- [[inter-client-dependency|Inter-Client Dependency]]

[^src-fedhint]: [[source-fedhint]]
