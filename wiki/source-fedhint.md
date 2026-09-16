---
title: "FedHINT: Inter-Client Dependency Recovery with Hidden Global Components for Federated Traffic Prediction"
type: source-summary
tags:
  - federated-learning
  - traffic-forecasting
  - privacy-preserving
  - aaai-2026
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# FedHINT

Hang Zhou、Wentao Yu、Yang Wei、Guangyu Li（南京理工大学），Sha Xu（广东工业大学），Chen Gong（上海交通大学）。AAAI-26, pp. 28946–28954。代码 github.com/lichuan210/FedHINT。源文件 `raw/inter-client-dependency-recovery-federated-traffic-prediction-aaai26.pdf`。

## 问题

联邦设定下每个客户端只看得到局部子图，跨区域传感器相关性缺失，即 [[inter-client-dependency|missing inter-client dependency]][^src-fedhint]。论文按 Introduction 的叙述把既有工作归为三类：FedGRU、CTFL、FedTPS 不建模该依赖；CNFGNN 用服务器上的预定义图、FedGTP 用服务器聚合的中间特征建模，两者都需上传数据特征；FedGCN 估计缺失节点，论文认为 non-IID 下难以推断[^src-fedhint]。

## 方法

假设局部数据含反映跨区域变化的 [[hidden-global-components|隐藏全局成分]]，在本地生成 [[proxy-node-generation|代理节点]] 顶替缺失的跨区域节点[^src-fedhint]。

1. 共享 global queries $Q\in\mathbb{R}^{N\times d_{att}}$ 引导注意力；key/value 先经 [[time-shifted-filters|时移滤波]]（$L$ 个频域滤波器，$j=t\bmod L$），投影回时域得 $\mathbf{X}^{proxy}_t\in\mathbb{R}^{N\times T_1}$，与局部节点拼成 $X^{all}_t\in\mathbb{R}^{(|V_m|+N)\times T_1}$。
2. query 之间加正交正则 $L_{div}=\frac{1}{N(N-1)}\sum_{i<j}|q_i^\top q_j|$，总损失 $\mathcal{L}=\sum_t|\hat{x}_t-x_t|+\lambda L_{div}$，$\lambda=0.1$。
3. 空间编码器用 AGCRN 的自适应邻接 $\tilde{A}=I+\sigma(EE^\top)$；global/local 两张掩码把编码器分成两支，拼接后预测全部 $|V_m|+N$ 个节点，取前 $|V_m|$ 行。
4. 只聚合隐藏成分提取器与 global encoder 的参数，local encoder 与 predictor 留在本地（[[personalized-aggregation-strategy|PAS]]）。

## 实验

12 步输入预测 12 步；METIS 把全局路网切成 6 个子图作客户端；6:2:2 划分；每个客户端 2 轮本地训练、共 200 轮全局；Adam，学习率 0.003，batch 64；逐客户端评估后跨客户端平均[^src-fedhint]。

| 数据集 | FedHINT MAE / RMSE | 次优基线 MAE / RMSE |
|---|---|---|
| PEMS03 | 11.95 / 19.12 | 15.48 / 23.92（FedTPS） |
| PEMS04 | 16.11 / 26.39 | 19.65 / 31.21（FedTPS） |
| PEMS07 | 17.19 / 28.42 | 21.54 / 34.27（FedGTP） |
| PEMS08 | 12.38 / 20.52 | 15.90 / 25.47（FedGTP） |

对照基线为 FedGRU、CNFGNN、CTFL、FedGCN、FedGTP、FedTPS[^src-fedhint]。消融（PEMS03，MAE/RMSE）：去代理节点 12.70/20.39、去时移滤波 12.11/19.28、去 global encoder 12.52/20.22、去个性化聚合 12.52/20.19，完整模型 11.95/19.12[^src-fedhint]。超参：$N=64$、$L=288$、$d_{att}=32$、$\lambda=0.1$[^src-fedhint]。

## 边界

- 隐私论证只靠「不上传数据特征」，没有差分隐私、安全聚合等机制，也没有攻击实验[^src-fedhint]。
- 摘要与结论称相对次优方法平均降 MAE 3.73、RMSE 4.81。按 Table 1 核算：MAE 降幅 3.53 / 3.54 / 4.35 / 3.52，均值 3.73，与自述一致；RMSE 降幅 4.80 / 4.82 / 5.85 / 4.95，均值 5.11，与自述不一致。论文未说明该平均值的口径[^src-fedhint]。
- 资源开销只给了 PEMS03 一图：通信量与单轮训练时间高于 FedGRU/CTFL/FedTPS，低于上传数据特征的 CNFGNN/FedGTP[^src-fedhint]。

[^src-fedhint]: [[source-fedhint]]
