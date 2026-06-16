---
title: "Cross-Gated Mechanism"
type: technique
tags:
  - gating
  - spatio-temporal
  - feature-interaction
  - traffic-imputation
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: medium
status: active
---

# Cross-Gated Mechanism (交叉门控机制)

Cross-Gated Mechanism 是 [[past|PAST]] 的 CGM（Cross-Gated Module）中提出的双向时空门控操作，用于从外部特征（时间戳、节点属性）中高效提取辅助时空模式[^src-past]。

## 动机

朴素做法是将空间嵌入向量 $\boldsymbol{v}_s \in \mathbb{R}^d$ 和时间嵌入向量 $\boldsymbol{v}_t \in \mathbb{R}^d$ 拼接后通过 $2d \times 2d$ 线性层映射。但此方法直接混合时空模式，未显式建模两者之间的交互细节，限制了拟合能力[^src-past]。

## 机制

Cross-Gated Layer 受 GLU (Gated Linear Unit) 启发，包含两步：

### Step 1: 特征投影 (Feature Projection)

用四个 $d \times d$ 线性子层（而非一个 $2d \times 2d$ 层）分别处理：

$$\begin{aligned} \boldsymbol{v}_{sp} &= W_{sp} \boldsymbol{v}_s, \quad \boldsymbol{v}_{tp} = W_{tp} \boldsymbol{v}_t \quad \text{(投影)} \\ \boldsymbol{v}_{sg} &= W_{sg} \boldsymbol{v}_s, \quad \boldsymbol{v}_{tg} = W_{tg} \boldsymbol{v}_t \quad \text{(门控)} \end{aligned}$$

参数总量仍为 $4d^2$，与简单拼接方案相同，但结构上分离了投影和门控路径。

### Step 2: 特征选择与跨域交互 (Feature Selection & Interaction)

门控向量先经 sigmoid 筛选本域特征，再经 tanh 模拟跨域正负关系：

$$\begin{aligned} \boldsymbol{v}_{sp} &\leftarrow \boldsymbol{v}_{sp} \cdot \text{Sigmoid}(\boldsymbol{v}_{sg}) \cdot \text{Tanh}(\boldsymbol{v}_{tg}) \\ \boldsymbol{v}_{tp} &\leftarrow \boldsymbol{v}_{tp} \cdot \text{Sigmoid}(\boldsymbol{v}_{tg}) \cdot \text{Tanh}(\boldsymbol{v}_{sg}) \end{aligned}$$

设计要点：
- **Sigmoid**：元素级门控，筛选本域投影中的相关特征、抑制无关特征
- **Tanh**：输出范围 $(-1, 1)$，可模拟空间和时间模式之间的正负交互
- **交叉结构**：空间投影经时间门控筛选（反之亦然），实现双向信息流通

最后经残差连接 $\boldsymbol{v}'_s = \boldsymbol{v}_s + \boldsymbol{v}_{sp}$ 防止梯度消失。

## 效率

标准线性层（输入 $2d$，输出 $2d$）参数量为 $4d^2$；cross-gated layer 同样使用四个 $d \times d$ 子层，参数量同为 $4d^2$，但通过分离门控路径和跨域交互，在不增加参数的前提下提升了表示能力[^src-past]。

## PAST 中的整体作用

CGM 由 $n$ 个 cross-gated layer 堆叠而成。每层将输出的隐向量（跨域交互结果的拼接）前传至 GIM 对应层进行跨模块信息交换。通过受 GBDT 启发的 ensemble 训练框架（CGM 拟合 GIM 的训练残差），交叉门控提取的辅助模式成为主模式的有效补充。

## 相关页面

- [[past]] — PAST 模型
- [[interval-aware-dropout]] — GIM 中的 dropout 机制
- [[primary-auxiliary-patterns]] — 主-辅模式概念

[^src-past]: [[source-past]]
