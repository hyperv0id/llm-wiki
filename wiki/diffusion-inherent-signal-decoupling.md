---
title: "Diffusion / Inherent Signal Decoupling（DSTF）"
type: technique
tags:
  - traffic-forecasting
  - decoupling
  - residual-decomposition
  - spatial-temporal
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Diffusion / Inherent Signal Decoupling（DSTF）

DSTF（Decoupled Spatial-Temporal Framework）主张把每条传感器时间序列显式拆成两类分量：从其它传感器扩散进来的 **diffusion signal**，以及与其它传感器无关的 **inherent signal**（本地产生的车辆）[^src-2206-09112-d2stgnn]。二者的分离没有监督信号，靠两个可学习的机制逼近：估计门给出扩散分量的比例，残差分解把「已解释的部分」从输入里扣掉、只把残余往下传。DSTF 是在 [[d2stgnn|D²STGNN]] 里提出的，实例如何选扩散块与固有块不在框架约束内。

## 与既有范式的差别

论文把既有 STGNN 的前提概括为「交通系统是扩散过程」：DCRNN 用扩散卷积替换 GRU 的全连接层，Graph WaveNet 逐层堆叠门控 TCN 与 GCN，两者都在扩散方向解释全部观测[^src-2206-09112-d2stgnn]。DSTF 的反对意见是这种解释把两类隐藏信号混在一起建模；更深的一层是 [[dcrnn|DCRNN]] 式的预定义邻接矩阵静态，而同一对节点的相互影响随时间变化[^src-2206-09112-d2stgnn]。

与时间序列里常见的趋势/季节分解相比，这里的分解轴不是「时间尺度」而是「空间来源」：拆出来的固有分量对应的是不经过邻居的那部分信号，周期性与趋势没有被显式建模[^src-2206-09112-d2stgnn]。

## 两个机制

### 估计门

$$\begin{aligned}
\Lambda_{t,i} &= \operatorname{Sigmoid}\Big(\sigma\big((\mathbf{T}^D_t\|\mathbf{T}^W_t\|\mathbf{E}^u_i\|\mathbf{E}^d_i)\mathbf{W}_1\big)\mathbf{W}_2\Big) && \text{// } [4d]\to[d]\to(0,1)\ \text{——每个（时刻，节点）一个标量}\\
\mathbf{X}^{dif} &= \Lambda\odot\mathbf{X}_l && \text{// } [T_h,N,1]\text{ 广播到 }[T_h,N,d]\ \text{——所有通道共用同一比例}
\end{aligned}$$

门的存在理由是「第一个块拿到的是原始数据，却只该学其中一部分」[^src-2206-09112-d2stgnn]。要注意门的输入只有日内时间槽嵌入、星期嵌入与 source/target 节点嵌入，并不读取当前信号内容——它给的是「这个时刻这个节点的经验扩散占比」，不是逐样本的即时判断[^src-2206-09112-d2stgnn]。

### 残差分解

$$\begin{aligned}
\mathbf{X}^{inh} &= \mathbf{X}_l-\mathbf{X}_b^{dif} && \text{// backcast 重建扩散块能解释的部分，从输入扣掉}\\
\mathbf{X}^{l+1} &= \mathbf{X}^{inh}-\mathbf{X}_b^{inh} && \text{// 固有块同样 backcast，剩余进入下一层}\\
\mathbf{X}_b^{*} &= \sigma(\mathbf{H}^{*}\mathbf{W}_b^{*}) && \text{// 两层非线性全连接，}\sigma=\operatorname{ReLU}
\end{aligned}$$

backcast 分支按信息重构方式学习，残差链接负责「扣掉已解释、留下未分解」[^src-2206-09112-d2stgnn]。顺序上先扩散后固有是论文的写法，但两支路原则上可交换[^src-2206-09112-d2stgnn]。

## 证据

分解没有真值，论文用下游预测精度间接验证，共三条证据链[^src-2206-09112-d2stgnn]。

- 解耦 vs 耦合（Table 4）：把动态图固定成静态图后，保留 DSTF 的 D²STGNN† 在 METR-LA h12 MAE 上 3.38，去掉门与残差分解的 D²STGNN‡ 为 3.51，同一行里 GWNet 3.53、DGCRN† 3.64；PEMS04 上差距最大，19.59 vs 22.57[^src-2206-09112-d2stgnn]。论文据此认为解耦本身（而非动态图）是主要收益来源，耦合版本相对基线优势有限[^src-2206-09112-d2stgnn]。
- 单独去掉门或残差（Table 5，METR-LA h12 MAE）：完整 3.35，`w/o gate` 3.44，`w/o res` 3.37，两者同时去掉 3.51[^src-2206-09112-d2stgnn]。门单独缺失的退化大于残差单独缺失，与「第一个块需要先知道自己该吃多少」的说法一致。
- 顺序可交换（Table 5 的 `switch`）：把固有块放到扩散块之前，h12 MAE 3.35，与完整模型持平[^src-2206-09112-d2stgnn]。

## 边界

- 两类隐藏信号都没有监督真值，$\Lambda$ 的分布、分解出来的两条序列长什么样，论文没有报告；分解是否真的对应物理上的「本地车流 / 流入车流」无法从实验外部核验[^src-2206-09112-d2stgnn]。
- 门由嵌入决定而非由输入决定，意味着同一节点在同一时间槽上的扩散占比在整个测试期是固定的；若某天的实际流入结构与经验占比偏离，门无法自我修正[^src-2206-09112-d2stgnn][INFERENCE]。
- 框架把扩散块与固有块的具体结构留作自由项，论文只给了一个实例（D²STGNN），没有独立的框架级消融来解释「为什么是这两个块」[^src-2206-09112-d2stgnn]。

## 相关页面

- [[d2stgnn]] — DSTF 的实例
- [[spatial-temporal-localized-convolution]] — 实例使用的扩散块
- [[source-2206-09112-d2stgnn]]
- [[traffic-forecasting]]

[^src-2206-09112-d2stgnn]: [[source-2206-09112-d2stgnn]]
