---
title: "LWR 交通流偏微分方程"
type: technique
tags:
  - traffic-flow
  - pde
  - conservation-law
  - physics-informed
created: 2026-09-16
last_updated: 2026-09-16
source_count: 2
confidence: medium
status: active
---

# LWR 交通流偏微分方程

Lighthill-Whitham（1955）与 Richards（1956）提出的宏观交通流模型：把车流视为连续介质，用运动波描述拥挤车队的演化，核心是密度守恒[^src-gencast]。

## 方程

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho x)}{\partial l} = 0$$

$\rho=\rho(l,t)$ 为密度，$x=x(l,t)$ 为速度。方程含 $\rho$ 与 $x$ 两个未知量，需补本构关系才封闭[^src-gencast]。

## 速度形式

交通预测数据集通常只给速度不给密度，故论文采用 Greenshields 线性关系 $x = x_{fspd}(1-\rho/\rho_{max})$（$x_{fspd}$ 自由流速度，$\rho_{max}$ 最大密度），消去 $\rho$ 得[^src-gencast]

$$\frac{\partial x}{\partial t} + (2x - x_{fspd})\frac{\partial x}{\partial l} = 0,\qquad
R = \frac{\partial x}{\partial t} + (2x - x_{fspd})\frac{\partial x}{\partial l}$$

## 两类用法

| 用法 | 做法 | 代表 |
|---|---|---|
| 软约束 | 用自动微分对网络预测求时空偏导，把残差 $R$ 作为损失项 | [[source-gencast\|GenCast]] |
| 硬约束 | 把守恒律离散为图算子：平流算子反对称（保能量）、扩散算子半正定（产熵） | [[vehicle-centric-graph-traffic-pde\|MMCKM]] |

GenCast 的实现要点：以 $\hat X$（掩码图上的速度预测）代入 $x$，对可微时空嵌入求导得 $R=\partial\hat X/\partial TE_{enc}+(2\hat X-X_{fspd})\odot\partial\hat X/\partial L_{enc}$，用 $\mathrm{Huber}(R,\delta)$ 惩罚；$\delta$ 在热身一轮后取残差的 $\tau$-分位数固定，$\tau=100\%$ 时退化为 RMSE[^src-gencast]。

## 边界

- 运动波近似不含加速度动力学，不刻画换道、信号控制等微观行为；MMCKM 正是为保留车辆级扰动而改用车辆为节点的拉格朗日离散[^src-mmckm]。
- Greenshields 线性本构是简化，GenCast 未在极端拥堵或振荡波场景下给出该关系误差的分析；论文把物理项定位为泛化正则，而非替代数据拟合[^src-gencast]。

## 相关页面

- [[physics-informed-neural-network|PINN]] — 损失约束型与架构嵌入型
- [[unobserved-region-forecasting|无观测区域交通预测]]

[^src-gencast]: [[source-gencast]]
[^src-mmckm]: [[source-mmckm]]
