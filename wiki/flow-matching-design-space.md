---
title: "Flow Matching Design Space"
type: concept
tags:
  - flow-matching
  - design-space
  - generative-model
  - tutorial
  - arxiv-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 5
confidence: high
status: active
---

# Flow Matching Design Space（FM 设计选择体系）

[[source-flow-matching-guide|FM 指南]]（Lipman et al., arXiv:2412.06264，2024-12）把 Flow Matching 组织为一个显式的设计选择体系（指南图 2 的 blueprint）：(a) 确定 source 分布 $p$ 与 target 分布 $q$；(b) 设计插值概率路径 $p_t$；(c) 用回归训练生成 $p_t$ 的 velocity field $u^\theta_t$；(d) 数值积分 ODE 采样。指南认为该 recipe 在 Riemannian 流形、离散空间 CTMC 与一般 CTMP 上保持不变，只替换每一步的具体对象[^src-flow-matching-guide]。本页按指南口径归置各设计轴，并挂接本 wiki 的具体方法页。

## 数学主干：Marginalization Trick 与 Bregman 散度

- 指南将"条件对象生成条件路径、边缘速度生成边缘路径"的定理命名为 **Marginalization Trick**（指南定理 3），并把条件化变量推广为任意 RV $Z$：边缘速度是条件速度按后验 $p_{Z|t}(z|x)$ 的加权平均。指南指出 FM 原论文只使用 $Z=X_1$ 的情形，一般 $Z$ 的表述引自 Tong et al. (2023)[^src-flow-matching-guide]。
- FM 损失与 CFM 损失梯度相等（指南定理 4）在指南中被推广为 **Bregman 散度学习条件期望** 的一般命题（命题 1）：梯度相等源于 Bregman 散度对第二变元的仿射不变性（归因 Holderrieth et al., 2024）。指南以此把 FM 家族全部可扩展损失——速度回归、$x_1$/$x_0$-预测、离散 FM、Generator Matching——统一为同一 Bregman 散度结构的特例[^src-flow-matching-guide]。
- 时间采样分布：指南转述 Esser et al. (2024) 的报告——直接从分布 $\omega(t)$ 采样 $t$ 等价于加权目标，但在大规模图像生成中前者实测更好[^src-flow-matching-guide]。

## 路径设计

- **条件流构造**：路径由条件流 $\psi_t(x|x_1)$（$t=0$ 为恒等、$t=1$ 映到 $x_1$ 的 diffeomorphism）经 push-forward 与求导同时给出条件路径与条件速度（指南式 4.28-4.31），这是指南给出的统一构造入口[^src-flow-matching-guide]。
- **affine 条件流与 Gaussian 路径**：$\psi_t=\alpha_t x_1+\sigma_t x$（scheduler $(\alpha_t,\sigma_t)$）；独立耦合 + Gaussian 源给出条件 Gaussian 路径 $\mathcal{N}(\alpha_t x_1,\sigma_t^2 I)$，指南指出其按边缘概率覆盖 VP/VE 扩散路径（但扩散中 $p_0$ 只是近似 Gaussian，与 FM 路径精确满足边界条件不同）[^src-flow-matching-guide]。
- **线性 conditional OT 路径**：$\psi_t=(1-t)x+t x_1$。指南的变分刻画：它在所有条件流中最小化 kinetic energy 的一个上界（Jensen 不等式给出的界），当 target 退化为单点时即为动态 OT 的解析解；直线路径下目标样本可被单步 Euler 精确求解[^src-flow-matching-guide]。
- **流形**：affine 组合在流形上无定义，指南用 geodesic 条件流（$\exp_{x_0}(\kappa(t)\log_{x_0}x_1)$）与更一般的 premetric 条件流（$d(\psi_t,x_1)=\bar\kappa(t)d(x_0,x_1)$）替代；geodesic 有闭式 $\exp/\log$ 映射时仿真无关，premetric 路径则需训练内模拟（指南第 5.6 节）[^src-flow-matching-guide]。
- **离散**：离散状态空间用 factorized 路径与 mixture 路径 $\kappa_t\delta_{x_1}+(1-\kappa_t)\delta_{x_0}$（指南第 7.5 节）[^src-flow-matching-guide]。

## 数据耦合

指南把耦合 $(X_0,X_1)\sim\pi_{0,1}$ 列为独立设计轴：独立耦合 $\pi=p\,q$；配对数据的依赖耦合（指南给出的实例为图像超分辨率、in-painting 与去模糊，经 $\pi_{0|1}$ 加噪采样）；以及 **multisample couplings**（Pooladian et al., 2023；Tong et al., 2023）——在每批 $k$ 个样本上解 doubly stochastic 匹配以隐式构造非独立耦合，指南转述其结果：传输代价低于独立耦合，二次代价下 $k\to\infty$ 逼近 OT 并诱导直轨迹[^src-flow-matching-guide]。

## 速度参数化

对 affine 路径，指南给出 velocity / $x_1$-prediction / $x_0$-prediction / score 四种参数化的闭式转换表（指南 Table 1）；高斯路径下 score $=-x_{0|t}/\sigma_t$，与 $x_0$-prediction 成比例（指南指出扩散文献的 noise-prediction/ε-prediction 就是 $x_0$-prediction，只是记名不同）。指南同时指出：$x_1$-prediction 在 $t\to1$、$x_0$-prediction 在 $t\to0$ 处系数发散，理论上为可去奇点、实践需端点解析式处理；所有 scheduler 在 $t=1$ 理论上给出相同采样结果，且 affine 路径支持训练后 scheduler 变换（scale-time 变换）。详见 [[x-prediction]][^src-flow-matching-guide]。

## 条件化与引导

指南把条件化归为三类：$Z=X_1$（FM 原论文）、$Z=X_0$（归因 Esser et al., 2024）、双侧 $Z=(X_0,X_1)$（stochastic interpolants / rectified flow 的形式，归因 Albergo & Vanden-Eijnden 2022、Liu et al. 2022 等），并给出三者的等价条件——当条件流在两侧变元上均为 diffeomorphism 时，三种条件化给出相同的边缘速度；指南强调单纯插值（即使 $C^2$ 光滑）不足以保证边缘速度生成边缘路径，需 SI/RF 原文的额外条件[^src-flow-matching-guide][^src-stochasticinterpolants][^src-rectified-flow]。引导方面，指南把 classifier guidance 与 CFG 统一到条件/无条件 score 关系式上，并给出 CFG 的速度场形式 $\tilde u=(1-w)u(\cdot|\varnothing)+w\,u(\cdot|y)$；详见 [[classifier-free-guidance]][^src-flow-matching-guide]。

## 与扩散模型的关系

指南第 10 章的归类：扩散训练等价于在 Gaussian 路径 + 独立耦合 + $x_0$/score 再参数化 + 时间约定反转下的 FM 训练（Denoising Score Matching 损失即 $x_0$-prediction 的 CM 损失）；确定性采样（probability flow ODE）与 FM 的 ODE 采样相同，随机 SDE 采样等价于在该 ODE 上叠加 divergence-free 的 Langevin 分量。详见 [[probability-flow-ode]] 与 [[generator-matching]][^src-flow-matching-guide][^src-sde]。

## 相关页面

- [[flow-matching]] — FM 原论文（NeurIPS 2023）
- [[generator-matching]] — 指南第 8-9 章的 CTMP 统一框架
- [[rectified-flow]] — reflow 路线（耦合轴上的迭代拉直）
- [[interflow]] / [[stochastic-interpolant]] — 双侧条件化路线
- [[x-prediction]] — 参数化转换与流形性质
- [[classifier-free-guidance]] — CFG 及其 FM 形式
- [[probability-flow-ode]] — 概率流 ODE
- [[tsflow]] — multisample couplings 在时序预测中的使用
- [[optimal-transport]] — OT 背景
- [[source-flow-matching-guide]] — 指南 source-summary

[^src-flow-matching-guide]: [[source-flow-matching-guide]]
[^src-flow-matching]: [[source-flow-matching]]
[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
[^src-rectified-flow]: [[source-rectified-flow]]
[^src-sde]: [[source-sde]]
