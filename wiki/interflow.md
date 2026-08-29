---
title: "InterFlow"
type: technique
tags:
  - interflow
  - stochastic-interpolant
  - continuous-normalizing-flows
  - generative-model
  - simulation-free
  - iclr-2023
created: 2026-07-13
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# InterFlow

**InterFlow** 是 [[source-stochasticinterpolants|Albergo & Vanden-Eijnden (ICLR 2023)]] 基于 [[stochastic-interpolant|stochastic interpolant]] 构建的连续时间归一化流生成模型：用二次目标从样本学习概率流速度，再以 ODE 积分完成采样与似然估计，**训练无需对 ODE 求解器反向传播**[^src-stochasticinterpolants]。

## 方法管线

1. **选插值** $I_t$（默认三角插值；也可线性 $a_t x_0+b_t x_1$ 或可学习 Fourier 系数）[^src-stochasticinterpolants]。
2. **采样** $t\sim\mathrm{Unif}[0,1]$（实践中常用 Beta 重加权，使靠近目标时训练更充分）、$x_0\sim\rho_0$、$x_1\sim\rho_1$ 独立，构造 $I_t$ 与 $\partial_t I_t$[^src-stochasticinterpolants]。
3. **最小化** 经验二次目标
   $$
   G_{N,n,K}(\hat v)=\frac1{KnN}\sum_{k,i,j}\Big(|\hat v_{t_k}(I_{t_k}(x_0^i,x_1^j))|^2-2\,\partial_t I_{t_k}\cdot\hat v_{t_k}(I_{t_k})\Big).
   $$
4. **采样 / 似然**：用 Dormand–Prince 等 ODE 求解器积分 $\dot X=\hat v_t(X)$；似然由瞬时变化公式（CNF 迹估计）给出[^src-stochasticinterpolants]。
5. **（可选）优化传输**：对 $I_t$ 或 Gaussian 基参数 max $\min G$，缩短路径长度、提升似然（棋盘格实验）[^src-stochasticinterpolants]。

收敛诊断：监控 $\tilde G(\hat v)=G(\hat v)+\mathbb{E}[|\hat v_t(I_t)|^2]$ 是否逼近 0；Proposition 3 保证 $H(\hat v)$ 控制终点 $W_2$[^src-stochasticinterpolants]。

## 相对基线的定位

| 对比 | InterFlow 特点 |
|------|----------------|
| FFJORD / MLE CNF | 仿真无关二次损失；每 epoch 成本近似常数；MiniBooNE 约 400× 加速[^src-stochasticinterpolants] |
| Score-SDE / DDPM | 有限时间；任意两端密度；直接学 ODE 速度而非 score；Gaussian base 时速度可导出 score[^src-stochasticinterpolants] |
| [[flow-matching\|Flow Matching]] | 同期；SI 以任意 $I_t$ 固定路径再回归速度，并给出 max-min→OT 理论[^src-stochasticinterpolants] |
| [[rectified-flow\|Rectified Flow]] | 同期直线/ reflow 路线；SI 论文指出 reflow 对非精确映射敏感[^src-stochasticinterpolants] |

## 指南口径：双侧条件化的归类与警示

[[source-flow-matching-guide|FM 指南]]（arXiv:2412.06264，2024-12）把 stochastic interpolant 归类为双侧条件化 $Z=(X_0,X_1)$ 的条件流构造（与 $Z=X_1$ 的 FM、$Z=X_0$ 的形式并列），并给出两点口径：其一，当条件流在两个变元上均为 diffeomorphism 时，三种条件化给出相同的边缘速度与边缘路径；其二，单纯插值映射（即使 $C^2$ 光滑，指南给出 ReLU 型插值反例）不足以保证边缘速度生成边缘路径——需要 SI 原文（Albergo & Vanden-Eijnden, 2022）与 Liu et al. (2022) 补充的条件，指南认为这些条件比其定理 3（Marginalization Trick）的条件更难验证[^src-flow-matching-guide]。指南另转述 Shi et al. (2023) 与 Liu et al. (2023)：线性版 FM 可视为 bridge matching 的某种极限情形[^src-flow-matching-guide]。

## 实验结果（论文报告）

- **2D**：曲线/8-Gaussian/棋盘格模式分离良好；支持未知解析密度之间的数据集插值[^src-stochasticinterpolants]。
- **表格 NLL**（越低越好）：相对 MADE/RealNVP/Glow/CPF/NSP/FFJORD/OT-Flow，在 POWER/GAS/HEPMASS/MINIBOONE 上达到更好或相当；BSDS300 略逊 FFJORD（约 0.6%）[^src-stochasticinterpolants]。
- **图像**（无数据增广）：CIFAR-10 NLL 2.99 BPD、FID 10.27；ImageNet $32\times32$ NLL 3.45、FID 8.49；Oxford Flowers $128\times128$ 展示 ab-initio ODE 流可扩展到此前 MLE 流难以达到的分辨率[^src-stochasticinterpolants]。

## 实现要点

- 速度网络：表格为 MLP（ReLU/ELU）；图像为 U-Net（DDPM 风格，正弦时间嵌入）[^src-stochasticinterpolants]。
- 时间采样 Beta$(\alpha,\beta)$ 重加权不改变理论极小点（任意 $\omega(t)>0$）[^src-stochasticinterpolants]。
- 可选正则 $\lambda\mathbb{E}[\|\nabla\hat v_t(I_t)\|^2]$ 控制 Lipschitz，论文主实验未启用[^src-stochasticinterpolants]。

## 局限

- 固定插值时传输一般非最优；学习 $I_t/\rho_0$ 的完整 OT 能力未充分规模化验证[^src-stochasticinterpolants]。
- 图像 FID 落后最强扩散（工程技巧较少）[^src-stochasticinterpolants]。
- 高维迹估计似然仍依赖 Hutchinson 类方法（与其他 CNF 相同）[^src-stochasticinterpolants]。

## 链接

- [[source-stochasticinterpolants]] — 论文 source-summary
- [[stochastic-interpolant]] — 理论概念
- [[flow-matching]] — 条件流匹配
- [[rectified-flow]] — Rectified Flow
- [[continuous-normalizing-flow]] — CNF
- [[optimal-transport]] — 最优传输
- [[benamou-brenier-algorithm]] — 动态 OT
- [[score-based-sde]] — Score-SDE 对照

[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
[^src-flow-matching-guide]: [[source-flow-matching-guide]]
