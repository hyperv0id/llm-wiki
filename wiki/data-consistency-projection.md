---
title: "Data-Consistency Projection"
type: technique
tags:
  - flow-matching
  - ode-solver
  - time-series-imputation
  - measurement-constraints
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# Data-Consistency Projection（DC 投影）

**DC 投影**是 TG-MSFM（ICLR 2026）推理期的硬约束机制：ODE 每步积分后，把已知坐标覆写回 linear bridge 的精确值，使观测测量在整条轨迹上逐字保留[^src-tgmsfm]。

## 机制

设 $z_1=\tilde{x}$（结构化端点）、$z_0$ 为初始噪声；已知索引集 $K=\{(t,d)\mid M_{t,d}=1\}\cup C$（观测数据坐标 ∪ 三个条件通道）。每个 Heun 步之后执行[^src-tgmsfm]：

$$z_{n+1}[K]\;\leftarrow\;(1-t_{n+1})\,z_0[K]+t_{n+1}\,z_1[K]$$

未观测坐标保持 ODE 演化。效果：已知坐标沿 linear bridge 精确运动，误差恒为零、不随步数积累；投影是线性操作 $O(|K|)$，相对 $O(LT^2Hd)$ 的骨干可忽略（附录 A.5）[^src-tgmsfm]。

## 论文给的两条性质（附录 A.3）

- **Prop A.1（oracle 速度下精确）**：若 $v_\theta\equiv z_1-z_0$（常数），Heun 对常数速度局部截断误差为零，每步精确恢复 $z_n=(1-t_n)z_0+t_n z_1$，DC 步零改动——观测坐标与插补坐标全部落在同一条桥上[^src-tgmsfm]
- **Prop A.2（非扩张）**：$P_K$ 是仿射子空间 $\{z: z[K]=\text{bridge}(t_{n+1})\}$ 上的正交投影，$\|P_K(u)-P_K(v)\|_2\le\|u-v\|_2$，等号当且仅当 $u-v$ 支撑在 $K$ 上——投影不放大误差[^src-tgmsfm]

## 在"观测约束注入方式"谱系中的位置

| 机制 | 约束方式 | 出处 |
|---|---|---|
| 训练期硬条件 | 观测值作为去噪网络输入 | [[csdi\|CSDI]] 类条件扩散 |
| 推理期软引导 | 以观测坐标的重构误差调制采样 | [[tsdiff\|TSDiff]] 的 [[observation-self-guidance]] |
| 前向过程条件化 | 观测值写入扩散前向转移并推导新 ELBO | [[rdpi\|RDPI]] 的 [[forward-process-conditioning]] |
| **每步 DC 钳制** | **积分后把已知坐标硬覆写回 linear bridge** | **TG-MSFM（ICLR 2026）** |

DC 的每步覆写在语义上最接近 inpainting 的硬约束，但作用于 FM 的 linear bridge 坐标而非扩散隐变量。论文消融（Table 2）显示 Euler→Heun 与 DC 的耦合在缺口边界收益最大：predictor-corrector 平均恰在 DC 约束观测坐标的位置降低局部截断误差，减少误差向相邻缺失时间戳的泄漏；toy 可视化（Fig B1，合成数据无训练）显示无 DC 的 Euler 在边界漂移、Heun+DC 保持在桥上并减少缺口内 ringing[^src-tgmsfm]。

## 适用边界

- 依赖 bridge 在 $K$ 上有闭式轨迹；换非线性 bridge 需重新推导钳制公式
- 只约束坐标值，不提供不确定性
- 每步执行一次，与积分步数线性相关（$O(N|K|)$）

## 相关页面

- [[tg-msfm]] — TG-MSFM 主页
- [[time-gated-multi-scale-velocity]] — 同文的另一半：时间门控多尺度速度场
- [[heun-sampler]] — Heun 二阶求解器（EDM 语境）
- [[tsflow]] / [[giflow]] / [[loft]] — 其他 FM 时序方法
- [[observation-self-guidance]] — 推理期软条件化对照（TSDiff）
- [[forward-process-conditioning]] — 前向过程条件化对照（RDPI）
- [[probability-flow-ode]] — 扩散侧的确定性 ODE 对照

[^src-tgmsfm]: [[source-time-gated-multi-scale-flow-matching]]
