---
title: "收缩估计的周期背景（Shrinkage Periodic Background）"
type: technique
tags:
  - periodicity
  - residual-learning
  - traffic-forecasting
  - decomposition
created: 2026-10-24
last_updated: 2026-10-24
source_count: 2
confidence: medium
status: active
---

# 收缩估计的周期背景

DeCoDy 的 Global and Local Residual Representation 用训练集统计量构造固定的周周期背景，历史与未来都表示为相对它的残差（式 1–2，§3.1）[^src-decody-iclr2027]。它对应「不同日期共享整体日 profile、局部峰谷幅度不同」这一现象（Fig. 1c）[^src-decody-iclr2027]。

## 背景的构造

对节点 $i$ 与周内槽位 $\tau$（一天 288 个 5 min 槽，一周 2016 槽）[^src-decody-iclr2027]：

$$P_i(\tau)=\alpha_{i,\tau}\,\mu^{week}_{i,\tau}+(1-\alpha_{i,\tau})\,\mu^{day}_{i,\tau\bmod 288},\qquad \alpha_{i,\tau}=\frac{n_{i,\tau}}{n_{i,\tau}+\lambda}\tag{1}$$

$\mu^{week}_{i,\tau}$ 是该周槽的训练均值，$\mu^{day}_{i,\tau\bmod 288}$ 是对应日槽的均值，$n_{i,\tau}$ 是该周槽的有效训练观测数，$\lambda$ 控制向日 profile 的收缩强度（默认 5，Table 4）[^src-decody-iclr2027]。观测少的周槽自动向样本更多的日槽靠拢，无需逐槽拟合。

历史残差同时充当动力学的初值边界：

$$r_i[t]=x_i[t]-P_i(\tau(t)),\qquad t=1-L,\dots,0\tag{2}$$

缺失值先用训练均值填充（标准化后为 0），掩码保留以区分「观测到的 0」；残差视为非观测数据，不进入观测 mask。周槽统计缺失时依次回退到日槽、节点均值、全局均值（§G.2）[^src-decody-iclr2027]。

## 与预测的关系

背景不参与训练更新：训练段一次性估计后固定，模型只学残差，输出时加回 $P^{f}$ 并反标准化（式 15，§3.4）[^src-decody-iclr2027]。长期周期结构由统计量承担，网络容量集中在局部偏差上。

与[[residual-cycle-forecasting|残差周期预测]]相比，周期项不是可学习参数而是收缩后的样本均值；与[[cyclenet|CycleNet]] 显式学习循环参数的做法不同，DeCoDy 的背景不随梯度更新，代价是它无法超出训练集统计所覆盖的形态[^src-cyclenet][^src-decody-iclr2027]。

## 证据

- 消融 C-PB 去掉周期背景及其条件，在 12 个 setting 上误差均上升；7 天预测的 MAE 由 26.769 升至 28.584（PEMS04）、23.945 升至 25.059（PEMS08）、9.634 升至 10.229（Seattle），即 +6.78% / +4.65% / +6.18%（§4.3，Table 2）[^src-decody-iclr2027]。
- 收缩强度 $\lambda\in\{1,2.5,5,7.5,10\}$ 的扫描中 $\lambda=5$ 最好，两侧变差；Seattle 对 $\lambda$ 不敏感（§4.4，Fig. 4，图内数值不可读）[^src-decody-iclr2027]。
- 案例（Fig. 5）：Seattle 单个节点 122–136 h lead 的累积流量，预测低于背景 14.8%，观测低于背景 17.2%，相差 2.4 个百分点（§4.5、§B.4）[^src-decody-iclr2027]。

## 边界

背景完全由训练段估计、测试期不更新，不适用于训练段未覆盖的季节或事件形态[^src-decody-iclr2027]。式(5) 中 $P^{f}_i[k-1]$ 在 $k=1$ 时的取值，论文未给显式约定[^src-decody-iclr2027]。

[^src-decody-iclr2027]: [[source-decody-iclr2027]]
[^src-cyclenet]: [[source-cyclenet]]
