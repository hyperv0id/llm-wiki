---
title: "Graph-Informed Prior (Adaptive Spatiotemporal Filtering)"
type: technique
tags:
  - graph-signal-processing
  - flow-matching
  - prior-distribution
  - spatio-temporal
  - icml-2026
created: 2026-08-29
last_updated: 2026-08-29
source_count: 3
confidence: medium
status: active
---

# Graph-Informed Prior（图信息先验）

**图信息先验**是 [[giflow|GiFlow]]（ICML 2026）提出的流匹配源分布构造方式：把可观测时空信号视为空间图与时间图笛卡尔积上的图信号，经自适应时空图滤波生成先验样本，替代问题无关的各向同性高斯先验，使源分布对齐目标分布、缩短生成路径[^src-giflow]。它与 [[gaussian-process-prior-flow-matching|TSFlow 的 GP 先验]]（参数化核函数）[^src-tsflow]和 [[low-rank-prior-estimation|LOFT 的低秩先验]]（神经参数化分解）[^src-loft]同属"信息先验替代高斯先验"路线；GiFlow 论文将 FM 先验选择的灵活性归因引至 Tong et al. (2024)[^src-giflow]。

## 问题：问题无关先验的代价

扩散与流匹配模型默认从 N(0,I) 出发，先验忽略数据的时空结构，与目标分布差异显著，模型必须学习更长的输运路径[^src-giflow]。GiFlow 利用 FM 源分布可任选的特点，用观测信号的图滤波平滑估计作为先验起点[^src-giflow]。

## 机制

### 乘积图上的时空滤波

设空间图拉普拉斯为 L_η、时间图拉普拉斯为 L_ξ。将 vec(X₁ᴹ) 视为乘积图上的图信号，联合滤波算子取 Kronecker 和 L_ηξ = τ_ξ L_ξ ⊕ τ_η L_η，滤波为指数算子 x_τ = e^{−L_ηξ} x₁ᴹ，矩阵形式[^src-giflow]：

$$X_\tau = e^{-\tau_\eta L_\eta} X_1^M e^{-\tau_\xi L_\xi}$$

指数滤波的 Taylor 展开是多项式滤波之和：对任意非零 (τη, τξ)，信息沿图传播到所有节点与时间步；实际实现截断到 K_η 个空间跳数与 K_ξ 个时间跳数[^src-giflow]。

### 滤波因子的自适应优化

(τη, τξ) 由如下最小化问题在训练数据上以 SGD 求解、推理时固定：第一项约束滤波结果与观测信号的对齐，第二项以 α_τ 加权鼓励拉普拉斯平滑（论文引 Bontonou et al. 2019、Dong et al. 2020 的图平滑损失）[^src-giflow]。

### 自适应时空感受野

**Proposition 3.1**：截断误差由 (τη C_s)^k / k! 与 (τξ C_t)^m / m! 的尾部和控制，其中 C_s、C_t 为两个拉普拉斯的谱半径[^src-giflow]。推论：滤波因子越小，感受野越局部，所需截断阶数越低；因子越大，感受野覆盖越远程依赖。实证上（Air-36/AQI，缺失率 20–60%），τη、τξ 随缺失率上升而增大；block missing 下 τη 增幅远大于 τξ——连续时间段缺失造成大时间空洞，模型更依赖空间滤波[^src-giflow]。

### 传输代价控制

**Theorem 3.2**：取 α_τ=0 时问题 (5) 的最优滤波因子构造先验 p₀ᴳ，则沿概率路径的期望二次传输代价满足 CFM(p₀ᴳ → q₁) ≤ CFM(p₀^Gauss → q₁)[^src-giflow]。直觉：高斯先验忽略时空结构，模型需要经过更长的路径；图信息先验通过空间平滑与时间一致性使源分布贴近目标[^src-giflow]。

### 线性条件流

条件路径取线性插值 ϕt(X|Z) = (1−t)·e^{−τηLη}X₁ᴹe^{−τξLξ} + t·X₁，对应向量场 u_t = X₁ − X_τ；论文称线性条件路径在动能误差界意义下最优（引 Lipman et al. 2023）[^src-giflow]。先验是观测的确定函数，故推理为确定性 ODE 积分、无需多次采样平均；需要不确定性量化时可向先验注入高斯噪声[^src-giflow]。

## 证据

- 先验消融（Air-36 point 20%，Table 4）：传输代价 FM-Gauss 299.62 → TFM（仅时间）123.39 → GFM（仅空间）115.05 → GiFlow 104.29，MAE 相应 12.79 → 10.12 → 9.75 → 9.54；仅空间滤波的 GFM 已优于 Table 2 全部基线[^src-giflow]。
- 感受野自适应（Fig 3/5）：滤波因子随缺失率增大，与 Proposition 3.1 的分析一致[^src-giflow]。
- 滤波因子优化开销可控（Table 8）：Air-36 0.19 分钟，5 万节点合成图 69.85 分钟 / 19.88 GB（100 epochs，batch 64）[^src-giflow]。

## 范围与边界

- Theorem 3.2 的条件是 α_τ=0 的最优滤波因子，实验中 α_τ 为验证集调参项（0.1–0.0001）；定理结论限于期望二次传输代价，与下游插补精度仅在消融中经验相关[^src-giflow]。
- 滤波因子优化依赖含完整真值的训练数据[^src-giflow]。
- 依赖图结构输入：Air-36/AQI 的空间图由训练数据经高斯核 + 阈值二值化构造，阈值 0.02 与 0.6 的极端图上性能明显退化[^src-giflow]。

[^src-giflow]: [[source-giflow]]
[^src-tsflow]: [[source-tsflow]]
[^src-loft]: [[source-loft]]
