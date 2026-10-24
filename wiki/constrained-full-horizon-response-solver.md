---
title: "受约束的全 horizon 响应求解器（Constrained Full-Horizon Response Solver）"
type: technique
tags:
  - fixed-point-iteration
  - neumann-series
  - error-control
  - long-horizon-forecasting
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# 受约束的全 horizon 响应求解器

DeCoDy 把未来残差的[[multi-delay-coupled-accumulation|延迟递推]]改写成作用在整个 horizon 上的线性系统，用并行不动点迭代求出全部未来位置，而非逐段自回归（式 11–14，§3.3）[^src-decody-iclr2027]。

## 求解的问题

$$r=r^{base}+\mathcal K(r),\qquad \mathcal K=\mathcal H\circ\mathcal M\tag{11}$$

$\mathcal H$ 是 $d_i$ 固定后折叠出的指数核因果卷积，$\mathcal M$ 是未来轨迹间的延迟图消息，两者都在 $[H,N]$ 上逐元素定义。等价方程组为 $(I-\mathcal K)r=r^{base}$；论文未把它与 ODE/SDE 求解器关联，全篇未出现 neural ODE 或离散化方法的表述[^src-decody-iclr2027]。

## 两个约束的分工

| 约束 | 形式 | 作用 |
|---|---|---|
| 正延迟 | $\ell_{\min}\ge 1$，$\mathcal D=\{1,2,3,6,12\}$ | 使 $\mathcal K$ 幂零，$\mathcal K^{Q}=0$，$Q=\lceil H/\ell_{\min}\rceil$；精确解是有限 Neumann 级数 $r^{\star}=\sum_{p=0}^{Q-1}\mathcal K^{p}r^{base}$（式 13） |
| 增益界 | $\dfrac{1}{d_i}\sum_{j}\sum_{\ell}|A_{ij\ell}[k]|<\rho<1$（式 7），默认 $\rho=0.8$ | 使 $\mathcal K$ 收缩，给出后验误差界并保证解唯一 |

幂零性决定解的形式，收缩性只决定何时可提前停：$\ell_{\min}=1$ 时 $Q=H$，理论上 $H$ 次更新即精确，实际以误差界提前终止（§C.2）[^src-decody-iclr2027]。

## 迭代与停止判据

$$r^{(m+1)}=r^{base}+\mathcal K\big(r^{(m)}\big),\quad r^{(0)}=r^{base},\qquad \|r^{(m)}-r^{\star}\|_\infty\le\frac{q}{1-q}\|r^{(m)}-r^{(m-1)}\|_\infty\tag{12, 14}$$

其中 $q=\max_{i,k}\frac{1}{d_i}\sum_j\sum_\ell|A_{ij\ell}[k]|$。实现把界换算回原始流量单位，每 4 次迭代检查一次（至少更新 4 次），界不超过 0.05 即停，上限 64 次（$\rho=0.9$ 时 128），触发上限则报告不收敛并终止（§3.3、§G.3、Algorithm 1）[^src-decody-iclr2027]。梯度穿过实际迭代步反传，周期背景与标准化统计固定（§3.4）[^src-decody-iclr2027]。

## 代价与实测

单样本复杂度为图消息 $O(EKH)$、节点局部响应 $O(NH\log H)$、总计 $O\big(J(EKH+NH\log H)\big)$，$J$ 为实际迭代数；显存为耦合系数 $O(EKH)$ 加响应与 FFT 工作区 $O(NH)$（式 22–24，附录 D）[^src-decody-iclr2027]。

PEMS08（112 节点）在 RTX 4060 Laptop、batch 4、FP32 下，只计响应求解的 5 次中位耗时（Table 5）[^src-decody-iclr2027]：

| horizon | 并行全 horizon | 串行逐步 | 比值 |
|---|---|---|---|
| $H=288$ | 13.50 ms | 163.15 ms | 12.09× |
| $H=864$ | 13.70 ms | 492.86 ms | 35.98× |
| $H=2016$ | 32.54 ms | 1136.20 ms | 34.92× |

两种方式最大差 0.0126 流量单位（Table 5）。容差 0.01 / 0.05 / 0.20 对应 16 / 12 / 12 次更新，后验界 0.001423 / 0.033589 / 0.033589，相对 FP64 参考的最大误差 0.000491 / 0.012615 / 0.012615（Table 6）[^src-decody-iclr2027]。论文声明这些比值只是同一方程两种解法的比较，不是端到端加速比，也不保证跨数据集或跨硬件可移植（附录 E）[^src-decody-iclr2027]。

## 边界

后验界只约束残差迭代的截断误差，不含 FFT 有限精度舍入（§C.2）[^src-decody-iclr2027]。不收敛频率与不同 $\rho$ 下的迭代数分布均未报告。

[^src-decody-iclr2027]: [[source-decody-iclr2027]]
