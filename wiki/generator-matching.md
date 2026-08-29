---
title: "Generator Matching"
type: concept
tags:
  - flow-matching
  - ctmp
  - generator-matching
  - generative-model
  - arxiv-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Generator Matching（生成器匹配）

**Generator Matching (GM)** 是 [[source-flow-matching-guide|FM 指南]]第 8-9 章阐述的统一框架（归因 Holderrieth et al., 2024）：把流（flow）、扩散（diffusion）、跳跃（jump）过程与离散 CTMC 统一为一般连续时间马尔可夫过程（CTMP）的生成模型，训练目标从"回归 velocity field"推广为"回归过程的 **generator**"。指南称 GM 统一了近年绝大多数生成模型，包括扩散模型、discrete diffusion 与此前各章的 FM 变体，并给出任意模态与跨模态建模的构造[^src-flow-matching-guide]。

## 从流到 CTMP：generator 与 KFE

- 流模型是 CTMP 中"确定性 + 速度场参数化"的特例；一般 CTMP 由转移核 $p_{t+h|t}$ 定义，其一阶近似由 **generator** $L_t$ 刻画（经 test function 作用定义：$\langle p_{t+h|t},f\rangle(x)=f(x)+h[L_tf](x)+o(h)$）。指南指出 generator 与 Feller 过程一一对应，正如 velocity field 与流一一对应[^src-flow-matching-guide]。
- 边缘概率的演化方程从 Continuity Equation 换成 **Kolmogorov Forward Equation (KFE)**：$\frac{d}{dt}\langle p_t,f\rangle=\langle p_t,L_tf\rangle$；"生成概率路径"的定义与连续情形同构[^src-flow-matching-guide]。
- 四类 generator 形式（指南 Table 2）：流 $L_tf=\nabla f\cdot u_t$；扩散 $L_tf=\frac12\sigma_t^2\cdot\nabla^2 f$；跳跃 $L_tf=\int(f(y)-f(x))Q_t(dy,x)$；离散空间 CTMC 为跳跃的 rate 矩阵形式 $L_tf=f^\top u_t$[^src-flow-matching-guide]。

## 普适刻画（universal characterization）

指南的定理 18（改编 Courrège 1965、von Waldenfels 1965，证明在 Holderrieth et al. 2024）：弱正则假设下 Feller 过程的 generator 在 $\mathbb{R}^d$ 上可分解为 **流 + 扩散 + 跳跃** 三分量之和，在有限离散空间上必为 CTMC。指南称由此穷尽了 $\mathbb{R}^d$ 与离散空间上 CTMP 生成模型的设计空间[^src-flow-matching-guide]。

## GM 训练目标

- **线性参数化**：$L_tf(x)=\langle Kf(x),F_t(x)\rangle_x$，只学习 $F_t$（速度、扩散系数、跳跃核、rate 都是 $F_t$ 的实例）。
- **GM 损失**：Bregman 散度回归 $F_t$；其条件版本（Conditional GM 损失）与边缘版本梯度相等（指南定理 20），且指南证明此性质**必须**用 Bregman 散度才能成立[^src-flow-matching-guide]。
- **General Marginalization Trick**（指南定理 19）：边缘 generator 由条件 generator 按后验期望得到；指南明言连续（定理 3）、流形（定理 10）、离散（定理 14）各章的 Marginalization Trick 都是该定理的特例[^src-flow-matching-guide]。
- 条件 generator 的求解示例：mixture 路径 $\kappa_t\delta_z+(1-\kappa_t)p$ 的 KFE 由"跳向 $z$"的跳跃过程解出：强度 $\lambda_t=\dot\kappa_t/(1-\kappa_t)$、跳跃分布 $J_t=\delta_z$；同一构造在 $\mathbb{R}^d$ 上给出欧氏空间的跳跃生成模型[^src-flow-matching-guide]。

## 组合模型（指南命题 3）

generator 是线性算子、KFE 是线性方程，因此解可以线性组合：

1. **Markov superposition**：$\alpha_1 L_t+\alpha_2 L'_t$（$\alpha_1,\alpha_2\ge0$，$\alpha_1+\alpha_2=1$）——例如跳跃 + 流的叠加构成 piecewise-deterministic Markov process；
2. **divergence-free 分量**：不改变边缘的分量可直接加入——指南指出 Langevin 动力学与 Metropolis-Hastings 的 generator 都是 divergence-free，故任何 GM 模型可任意叠加 MCMC 步；指南在第 10 章正是用它导出扩散模型的随机采样；
3. **predictor-corrector**：前向 generator 与后向 generator 的组合（$\alpha_1-\alpha_2=1$）[^src-flow-matching-guide]。

## 多模态

乘积状态空间 $S=S_1\times S_2$ 上的 GM 模型可经 factorized 条件概率路径"复用"单模态模型（如图像 + 文本 = 连续 FM + 离散 FM）；指南称该构造使多模态建模有原则性基础，并引 Campbell et al. (2024) 的多模态蛋白质生成为实例[^src-flow-matching-guide]。

## 与其他框架页的关系

- [[flow-matching]] 与 [[flow-matching-design-space]]：GM 是 FM recipe 在 CTMP 层面的推广；FM 的 Marginalization Trick 与梯度等价定理均是其特例[^src-flow-matching-guide]。
- [[diffusion-model]] / [[score-based-sde]]：扩散是 GM 特例——SDE 由扩散分量 generator 刻画，随机采样对应叠加 divergence-free Langevin 分量[^src-flow-matching-guide]。
- [[generative-vector-field]] / [[flux-matching]]：本 wiki 此后收录的"把得分推广为一般生成向量场"路线，与 GM 的 generator 视角同处"超越速度场参数化"的方向（wiki 层面的归类对照，非指南论断）。

[^src-flow-matching-guide]: [[source-flow-matching-guide]]
[^src-sde]: [[source-sde]]
