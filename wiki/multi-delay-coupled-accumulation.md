---
title: "多延迟耦合与累积（Multi-Delay Interaction and Accumulation）"
type: technique
tags:
  - delayed-interaction
  - graph-dynamics
  - temporal-accumulation
  - traffic-forecasting
created: 2026-10-24
last_updated: 2026-10-24
source_count: 2
confidence: medium
status: active
---

# 多延迟耦合与累积

DeCoDy 的核心递推把未来残差的变化写成三项之和：自状态持续、局部驱动、多个延迟档上的邻居影响（式 9–10，§3.3）[^src-decody-iclr2027]。对应现象是邻接节点的偏差起始时刻错开、持续时间不同，并在后续时段重叠累积（Fig. 1a）[^src-decody-iclr2027]。

## 递推形式

$$r_i[k]=a_i\,r_i[k-1]+b_i\Big(s_i[k]+\sum_{j\in\mathcal{N}_i}\sum_{\ell\in\mathcal{D}}A_{ij\ell}[k]\,r_j[k-\ell]\Big),\qquad a_i=e^{-d_i},\quad b_i=\frac{1-e^{-d_i}}{d_i}\tag{9–10}$$

| 项 | 含义 | 参数来源 |
|---|---|---|
| $a_i r_i[k-1]$ | self-state persistence，节点自身状态的衰减延续 | $d_i>0$，horizon 内固定 |
| $b_i s_i[k]$ | local driving，相位条件化的驱动信号 | $s_i[k]$ 逐时步变化 |
| $b_i\sum_j\sum_\ell A_{ij\ell}[k]r_j[k-\ell]$ | delayed neighboring influence，邻居在多个延迟档上的影响 | $A_{ij\ell}[k]$ 逐时步、逐延迟变化 |

延迟集合固定为 $\mathcal{D}=\{1,2,3,6,12\}$，在 5 min 采样下对应 5–60 min，不随输入变化，也不是连续分布的学习对象（§3.3、Table 4）[^src-decody-iclr2027]。$a_i,b_i$ 与单位步长零阶保持下 $\dot r=-d\,r+d\,u$ 的离散系数同形，但论文未写连续时间方程、未用 ODE 求解器，此处只记系数形状[INFERENCE]。

与 [[pdformer|PDFormer]] 不同：PDFormer 用聚类出的短时交通模式变换注意力 key，交互仍即时；DeCoDy 的延迟进入递推的邻居项下标 $\ell$，影响跨时间步累积[^src-pdformer-jiang-2023][^src-decody-iclr2027]。

## 累积为何不发散

耦合经过尺度化归一（见 [[decody]] 式 7），使每个目标节点在一行内的总增益满足

$$\frac{1}{d_i}\sum_{j\in\mathcal{N}_i}\sum_{\ell\in\mathcal{D}}|A_{ij\ell}[k]|<\rho<1$$

在此约束下，命题 1 给出与 horizon 无关的上界（式 8，§3.3、附录 C.1）[^src-decody-iclr2027]：

$$R_0=\max_{i,t\le 0}|r_i[t]|,\quad U=\max_{i,1\le k\le H}\Big|\frac{s_i[k]}{d_i}\Big|\ \Longrightarrow\ \max_{i,1\le k\le H}|r_i[k]|\le\max\Big\{R_0,\ \frac{U}{1-\rho}\Big\}$$

证明只用行绝对和界，不要求系数非负或时不变；默认 $\rho=0.8$（Table 4）[^src-decody-iclr2027]。该上界只保证响应幅度不随 $H$ 显式增长，不保证预测精度。

## 与求解器的分工

$d_i$ 在 horizon 内固定，递推中 $r_i[k-1]$ 一支可解析求和为指数核 $\mathcal{H}_i$；跨节点延迟耦合 $\mathcal{M}$ 才需要迭代，两者合记为 $\mathcal{K}=\mathcal{H}\circ\mathcal{M}$，交由[[constrained-full-horizon-response-solver|全 horizon 求解器]]并行求解（§3.3）[^src-decody-iclr2027]。

消融 C-SD 把 $\mathcal{D}$ 收缩为单一 5 min 延迟，Table 2 的 12 个 setting 上 MAE 与 RMSE 都劣于完整模型，7 天预测在 Seattle 的 MAE 由 9.634 升至 10.584（+9.86%）（§4.3）[^src-decody-iclr2027]。论文未做 1 到 5 档的扫描，「多档优于单档」只有这一个二值对照。

[^src-decody-iclr2027]: [[source-decody-iclr2027]]
[^src-pdformer-jiang-2023]: [[source-pdformer-jiang-2023]]
