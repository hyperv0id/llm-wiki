---
title: "Intersection Trajectory Prediction"
type: concept
tags:
  - trajectory-prediction
  - intersection
  - autonomous-driving
  - intelligent-transportation
created: 2026-10-20
last_updated: 2026-10-20
source_count: 2
confidence: medium
status: active
---

# Intersection Trajectory Prediction（路口轨迹预测）

**路口轨迹预测**指预测路口处多个道路使用者（机动车、非机动车、行人）未来若干时间步的位置。本页记录的口径来自 [[trajar|TrajAR]]：输入是历史轨迹特征，输出是目标使用者未来 $t_f$ 步的位置序列，交互使用者的历史同时进入模型[^src-trajar]。

## 形式化

论文的定义[^src-trajar]：

- 时刻 $t$ 的特征 $z^\nu_t$ = 位置 $\tau^\nu_t=(x^\nu_t,y^\nu_t)$ + 信号状态 $s_t\in\mathbb{R}^{N_s}$；使用者类型为 one-hot $a_\nu\in\mathbb{R}^{N_a}$。
- 交互使用者集合 $V_I$ 取距目标最近的 $N_R$ 个使用者（TrajAR 实验 $N_R=15$）。
- 输入 $H=\langle z^\nu_{t_0-t_h+1},\dots,z^\nu_{t_0}\rangle$（$\nu\in V_I\cup\{\nu_T\}$），输出 $\langle\tau^{\nu_T}_{t_0+1},\dots,\tau^{\nu_T}_{t_0+t_f}\rangle$。TrajAR 取 $t_h=t_f=32$。
- 评价指标沿用 MTP-GO 的一组：ADE、FDE、MR（Miss Rate）、APDE（Average Path Displacement Error），加上 step@8/16/24 的分步误差[^src-trajar]。

## 论文对路口场景的特殊性判断

- **多类使用者混行**：机动车、自行车、行人的运动学与行为差异大，交互模式异构；论文把高速公路匝道场景对照为「只有一种使用者、交互同质」[^src-trajar]。
- **隐式交互**：交互不只由距离决定。论文用图 1(a) 的 Scene #2 说明：左转车与直行车在近处必须让行，相邻但路径不冲突的车辆与行人可视为非交互；判断依赖运动趋势、类型、信号与路权规则（右行交通下转弯让直行、右转让左转、机动车让行人）[^src-trajar]。
- **长时预测更难**：horizon 拉长后，逐步自回归累积误差、并行输出整体偏置的问题被放大[^src-trajar]。

## 输出框架的三类取舍

论文图 1(b) 并列三种解码方式[^src-trajar]：

| 框架 | 生成方式 | 论文指出的问题 |
|---|---|---|
| 逐步自回归 | 一步一次前向 | 误差累积，后段偏差大 |
| 非自回归 | 全部步并行 | 缺少前步纠偏，整体偏置，可能违反运动学约束 |
| 多尺度自回归 | 每个自回归步生成一个分辨率的全部位置 | 论文的解法，见 [[scale-wise-autoregressive-decoding]] |

## 论文整理的既有方法谱系

论文把工作按场景分两类[^src-trajar]：

- **高速公路匝道/汇入**：关注换道与加减速，代表 CS-LSTM、PiP（social pooling）、WSiP（波叠加式 wave pooling）、C2F-TP（扩散式轨迹生成）、BAT（动态几何图）、HLTP（视觉感知池化 + 蒸馏）。论文对这些工作的批评是：只有单一车型、交互同质，且数据多采集于自由流条件，缺少拥堵/强交互场景的验证。
- **路口**：早期单使用者方法（LSTM 或物理模型）无法建模多使用者动力学；多使用者方法包括 GRIP++、LG-ODE（GNN 编码 + RNN/neural ODE 解码）、MTP-GO（混合密度网络 + Kalman 滤波做多峰概率输出）、FJMP（稀疏交互图上的 factorized joint prediction）、KI-GAN（把信号状态、车辆运动、空间关系分路编码 + attention pooling）。论文称这些方法对隐式交互的刻画仍不足，在拥堵路口的长时预测上受限。

## 数据集与评价口径的现状

TrajAR 使用的四个真实路口数据集是：SinD-Tianjin、SinD-Xi'an（中国天津、西安信号交叉口，10 Hz）与 InD-Bendplatz、InD-Frankenburg（德国路口，25 Hz，四类使用者）；按录制时间 6:2:2 划分[^src-trajar]。论文报告的主窗长是 32 步，但正文对下采样后间隔的说明（0.12 s）只与 25 Hz 的 InD 自洽；SinD 为 10 Hz，同倍下采样按本页核算应为约 0.3 s，论文未区分两组，因此不同数据集上 32 步对应的物理时长在文中不明确[^src-trajar]。表中 ADE/FDE 以 (e-2) 口径给出，正文未说明该缩放的换算方式[^src-trajar]。

## 本 wiki 的记录范围

本页的路口设定来自 TrajAR 一个来源；该论文对其余方法的描述属论文自述的文献综述，未逐年核对原文，因此谱系一节只用于定位，不作为方法优劣判断[^src-trajar]。消融与主结果数字集中在 [[trajar]]。

wiki 内还有两类与「轨迹」相关但任务不同的页面，为避免下次 ingest 误挂：

- **同一任务、不同轴向**：[[source-adaptraj]]（AdapTraj）也是多智能体轨迹预测，但它处理的是多源域泛化（训练与评测都接触不到目标域），观测 8 步预测 12 步，数据集为 ETH&UCY、L-CAS、SYI、SDD，与 TrajAR 的 SinD/InD 信号路口设定不同[^src-adaptraj]。两条线可以在「多使用者预测」这个任务下合并比较，但不是同一个 benchmark。
- **共享「轨迹」二字、任务不同**：[[source-smartraj2]]、[[source-transfertraj]] 是轨迹/区域表征学习与迁移，[[source-trajectory-dl-survey]] 是轨迹数据管理/挖掘综述；三者都不预测未来位置，不接入本页谱系。

## Related Pages

- [[trajar]] — 模型页
- [[source-trajar]] — 论文源摘要
- [[scale-wise-autoregressive-decoding]] — 输出框架中的多尺度自回归
- [[motion-moe]] — 编码侧的交互/属性建模
- [[source-adaptraj]] — 同任务（多智能体轨迹预测）、不同轴向（多源域泛化）
- [[traffic-forecasting]] — 邻近任务：路网级交通状态预测（站点/路段级，与个体轨迹预测不同）

[^src-trajar]: [[source-trajar]]
[^src-adaptraj]: [[source-adaptraj]]
