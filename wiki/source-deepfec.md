---
title: "DeepFEC: Energy Consumption Prediction under Real-World Driving Conditions for Smart Cities"
type: source-summary
tags:
  - energy-consumption
  - deep-learning
  - traffic-prediction
  - smart-cities
  - spatio-temporal
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# DeepFEC: Energy Consumption Prediction

## 概述

DeepFEC 是 Elmi 和 Tan 在 WWW '21 上提出的混合深度学习模型，用于在真实驾驶条件下预测城市道路网络的车辆能耗[^src-deepfec]。该模型将残差神经网络（ResNet）与双向 LSTM（Bi-LSTM）结合，分别捕捉能耗的空间和时间模式，并融合车辆特征、速度、天气和周期性等上下文信息，在密歇根州道路网络上的实验中显著优于 10 个基准方法。

## 问题定义

论文将城市道路网络建模为有向图，每个节点代表交叉口，每条有向链路代表一个路段[^src-deepfec]。能耗数据被组织为三维张量：能耗张量（路段 × 时间槽 × 能耗维度）、车辆张量（记录车辆类型、重量、发动机配置和排量）和速度张量。目标是基于前 N 个时间槽的数据，预测未来第 p 个时间槽所有路段的能耗。

## 核心方法：四组件混合架构

DeepFEC 的架构由四个主要组件构成[^src-deepfec]：

1. **车辆特征提取**：使用 embedding 层将静态分类变量（车辆类型 ICE/HEV/PHEV/EV、发动机配置、排量、重量）转换为稠密向量表示。Embedding 层在此扮演类似全连接层的角色，但更高效地处理分类变量。

2. **空间模式提取**：通过多层残差单元捕捉道路网络拓扑中的空间依赖。每个残差单元包含三个"Look-Up + 3D 卷积 + 批归一化"组合。Look-Up 操作将道路网络的邻接矩阵嵌入到能耗和速度张量中，3D 卷积使用 1 × D × 2 的滤波器强调相邻道路的空间特征。残差连接（ResNet）使网络可以堆叠到 100 层甚至 1000 层以上，捕捉远距离道路之间的依赖。

3. **时间模式提取**：在 K 个残差单元之后，将空间特征重塑为时间序列格式，输入 Bi-LSTM 层。Bi-LSTM 同时从前向和后向学习时间依赖，输出时空向量，其最后一个元素即为预测的能耗值。

4. **天气与周期性特征**：通过全连接层提取三种周期特征——时间间隔间周期（相邻时间槽的依赖）、日周期（同一时刻在不同工作日的相似性）和周期（同一时刻在相邻周的相似性）。天气信息（晴天/雨天/风等）通过 one-hot 编码后由全连接层处理。

最终，所有组件的输出通过 Tanh 激活函数和 Hadamard 积融合，产生预测值[^src-deepfec]。

## 实验与结果

论文在两个数据集上进行了评估[^src-deepfec]：

- **VED 数据集**：包含 366 辆车（249 辆 ICE、90 辆 HEV、24 辆 PHEV、3 辆 EV）在密歇根州一年的真实驾驶数据，总计 357,000 英里。数据包括车辆速度、能耗、发动机信号和环境温度，以及 GPS 轨迹。
- **SPMD 数据集**：约 3,000 名驾驶员在安娜堡的安全试点模型部署数据。

DeepFEC 在 VED 数据集上取得了 RMSE 0.474、MAE 0.290、MAPE 0.029 的成绩，显著优于 10 个基准方法[^src-deepfec]。消融实验表明，速度模式对预测精度的贡献最大（移除速度后 RMSE 升至 0.694），其次是周期性特征，天气特征的影响相对较小（因训练和测试数据来自同一季节）。DeepFEC 在约 40 个 epoch 后达到最佳精度，训练效率优于 LSTM、GRU 和 Bi-LSTM。

## 局限性

论文主要关注密歇根州的道路网络，在市中心区域的预测精度较低（RMSE 约 1.41），因为行人流量大、交通模式复杂[^src-deepfec]。电动车样本量极小（仅 3 辆），预测误差较高。未来方向包括分析交通信号灯对市中心区域能耗模式的影响。

[^src-deepfec]: [[source-deepfec]]