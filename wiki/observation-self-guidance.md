---
title: "Observation Self-Guidance"
type: technique
tags:
  - diffusion
  - guidance
  - time-series
  - conditional-generation
  - probabilistic-forecasting
created: 2026-07-13
last_updated: 2026-07-25
source_count: 2
confidence: medium
status: active
---

# Observation Self-Guidance

**Observation self-guidance** 是 [[tsdiff|TSDiff]]（NeurIPS 2023）提出的推理期条件化技术：在**不改训练、不引入辅助网络**的前提下，用无条件时序扩散模型自身的一步去噪估计来构造引导分布 $p(y_{\mathrm{obs}}\mid x_t)$，从而对任意观测子序列采样 $p(y_{\mathrm{ta}}\mid y_{\mathrm{obs}})$[^src-prs]。

## 动机

[[classifier-guidance|分类器引导]]需要噪声鲁棒辅助分类器；[[classifier-free-guidance|CFG]] 需要联合训练条件/无条件模型[^src-prs]。时序预测/插补中的“条件”是**部分观测时间步**而非离散类别：若每次掩码模式都条件训练，模型会失去任务无关性与无条件生成能力（对比 [[timegrad|TimeGrad]]、[[csdi|CSDI]]）[^src-prs]。Self-guidance 的直觉是：**学会完整序列的模型也应能近似评价部分序列**[^src-prs]。

## 数学形式

Bayes 分解条件得分[^src-prs]：

$$
\nabla_{x_t}\log p_\theta(x_t\mid y_{\mathrm{obs}})
=
\nabla_{x_t}\log p_\theta(x_t)
+
\nabla_{x_t}\log p_\theta(y_{\mathrm{obs}}\mid x_t)
$$

引导反向过程（尺度 $s$）[^src-prs]：

$$
x_{t-1}\sim\mathcal{N}\big(\mu_\theta(x_t,t)+s\sigma_t^2\nabla_{x_t}\log p_\theta(y_{\mathrm{obs}}\mid x_t),\,\sigma_t^2 I\big)
$$

用去噪网络 $\epsilon_\theta$ 的一步重构

$$
\hat y=f_\theta(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

参数化 $p(y_{\mathrm{obs}}\mid x_t)$，再对 $x_t$ 自动微分得到引导项[^src-prs]。

## 两种引导分布

| 变体 | 似然 | 得分含义 | 实证倾向 |
|------|------|----------|----------|
| **Mean Square (MS)** | $\mathcal{N}(y_{\mathrm{obs}}\mid\hat y,I)$ | 观测段 MSE | 可用但 CRPS 较弱 |
| **Quantile (Q)** | 非对称 Laplace / pinball，$\kappa\in(0,1)$ | 多分位对齐 | 更贴 CRPS，主结果更优 |

实践中 Q 引导在多个均匀分位上对多样本预测分别引导，以更好覆盖 CDF[^src-prs]。$s$ 在验证集网格搜索（MS 常用 $4/32$；Q 因数据集而异，约 1–8）[^src-prs]。

## 与相关引导技术的对比

| 技术 | 训练改动 | 额外网络 | 条件类型 |
|------|----------|----------|----------|
| [[classifier-guidance]] | 无（扩散侧） | 噪声鲁棒分类器 | 类别等 |
| [[classifier-free-guidance]] | 条件/无条件联合 | 无 | 训练时见过的条件 |
| **Observation self-guidance** | **无** | **无** | **任意观测时间子集** |
| [[tsflow\|TSFlow]] CPS+引导 | 无条件 CFM | 无 | 观测上下文（向量场修正） |

TSFlow 将同类“无条件 → 推理条件”思想迁移到流匹配，并用条件先验采样强化起点[^src-tsflow]。

## 能力与代价

- **能力**：标准预测；上下文随机/块缺失（RM、BM-B、BM-E）下的预测，同一检查点即可，无需为每种缺失重训[^src-prs]。
- **软约束**：不保证观测位硬匹配，对齐由 $s$ 控制[^src-prs]。
- **代价**：每步反向需对网络求 $\nabla_{x_t}$，推理慢于纯条件前向（Exchange 上 self-guidance ≈201s vs TSDiff-Cond ≈163s）[^src-prs]。可用更快 ODE/引导采样器缓解，但不改训练[^src-prs]。

## 相关页面

- [[tsdiff]] — 承载该技术的无条件时序扩散模型
- [[source-prs]] — 原始论文
- [[linear-predictive-score]] — 同文提出的合成数据指标
- [[classifier-guidance]] / [[classifier-free-guidance]] — 经典图像域引导
- [[source-csdi]] / [[csdi]] — 训练期条件化对照
- [[source-timegrad]] / [[timegrad]] — 条件 AR 扩散对照
- [[source-tsflow]] / [[tsflow]] — 流匹配上的后续无条件→条件桥接
- [[prediction-refinement]] — 扩散密度做先验的数据空间精炼（同文另一条推理方案）
- [[score-function]] — 得分函数基础

[^src-prs]: [[source-prs]]
[^src-tsflow]: [[source-tsflow]]
