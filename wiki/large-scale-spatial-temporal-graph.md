---
title: "Large-Scale Spatial-Temporal Graph Forecasting"
type: concept
tags:
  - spatial-temporal
  - computational-complexity
  - scalability
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
references:
  - [[source-fast-long-horizon-forecasting]]
confidence: high
status: active
---

# Large-Scale Spatial-Temporal Graph Forecasting

## 定义

大规模时空图预测指在具有数千至数万个节点（图结构传感器/区域）的网络上进行时间序列预测，通常需要**长视野**（多步至一周）预测。

典型场景：
- **交通网络**：加州高速网络 8600+ 检测器
- **能源网络**：城市级电表/燃气表
- **空气监测网络**：跨城市的传感器阵列

## 核心挑战

### 1. 计算复杂度

| 模块 | 复杂度 | 大规模场景问题 |
|------|---------|-----------------|
| 空间卷积 (GCN) | O(N²) | 8600 节点 = 74M 边计算 |
| 时间注意力 | O(T²) | 672 步 = 451k 计算 |
| 联合时空 | O(N²T + NT²) | 内存爆炸 |

### 2. 内存消耗

现有 STGNN 在 CA 数据集（8600 节点，672 步预测）上需要 **48GB GPU 内存**，大多数 GPU 无法训练。

### 3. 长视野预测精度退化

预测步数增加时：
- 错误累积传播
- 空间依赖衰减
- 时序模式复杂化

## FaST 的解决方案

[[source-fast-long-horizon-forecasting|FaST]] 提出两个核心技术：

1. **[[adaptive-graph-agent-attention|AGA-Att]]**: O(N·a) 空间复杂度（a = 32 ≪ N）
2. **[[mixture-of-experts|HA-MoE]]**: 时间压缩输入避免 T 步展开

## 现有方法分类

### 结构感知方法
- **稀疏聚合** (SGP): 利用图拓扑减少计算
- **邻居采样** (SAGDFN): 随机采样邻居
- **图划分** (PatchSTG): 分块处理

局限：依赖准确图结构、丢弃长程依赖

### 无结构方法
- **线性注意力** (BigST): 绕过成对节点计算
- **随机投影** (RPMixer): MLPs 融合空间特征
- **空间标识** (STID): 消除空间交互

局限：过度简化交互、引入噪声、丢失空间语义

### 异质性感知方法
- **HA-MoE**: 动态选择专家适配节点和时间异质性
- **代理注意力**: 用少量代理 token 捕获空间冗余

## 数据集基准

LargeST 数据集（KDD 2024）：
| 数据集 | 节点数 | 时间粒度 | 时间跨度 |
|--------|--------|----------|----------|
| SD | 716 | 15min | 2019全年 |
| GBA | 2,352 | 15min | 2019全年 |
| GLA | 3,834 | 15min | 2019全年 |
| CA | 8,600 | 15min | 2019全年 |

## 相关页面

- [[traffic-forecasting]] — 城市交通预测场景
- [[adaptive-graph-agent-attention|AGA-Att]] — 空间复杂度优化
- [[mixture-of-experts|MoE]] — 特征提取与时间压缩

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]