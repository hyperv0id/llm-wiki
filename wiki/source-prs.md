---
title: "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - self-guidance
  - unconditional-generation
  - neurips-2023
created: 2026-07-13
last_updated: 2026-07-25
source_count: 0
confidence: low
status: active
---

# Source: Predict, Refine, Synthesize (TSDiff / PRS)

**作者**: Marcel Kollovieh (TUM)\*, Abdul Fatir Ansari (AWS AI Labs)\*, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang Wang（\*共同一作）
**发表**: NeurIPS 2023
**arXiv**: [2307.11494](https://arxiv.org/abs/2307.11494)（v3, 2023-11-22）
**代码**: [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion)
**领域**: 概率时间序列预测 / 无条件扩散 + 推理期条件化

## 核心论点

此前时序扩散工作（[[timegrad|TimeGrad]]、[[csdi|CSDI]]、SSSD）几乎都是**任务专用条件模型**：训练时就把预测/插补掩码写进目标，因而失去无条件生成能力，也难以在未知缺失模式上复用。本文提出 **[[tsdiff|TSDiff]]**——按数据集训练的**无条件**时序扩散模型——并用 **[[observation-self-guidance|observation self-guidance]]** 在推理期把任意观测子序列 $y_{\mathrm{obs}}$ 注入反向过程，无需辅助网络、也不改训练目标。同一模型覆盖三条用例：**Predict**（概率预测）、**Refine**（精炼基预测器输出）、**Synthesize**（合成数据训练下游预测器）。

## 方法

### 架构（TSDiff）

- 基于 SSSD 对 DiffWave 的改写：残差块内用 **S4** 沿时间维建模，**Conv1×1** 沿通道维交换信息。
- 单变量长度 $L$；通过在通道维拼接 lag 特征扩展历史，输入 $x_t \in \mathbb{R}^{L \times C}$。
- 3 残差层 × 64 通道；扩散步 $T=100$，线性 $\beta_1=10^{-4},\beta_{100}=0.1$；GluonTS mean scaler 归一化。

### Observation Self-Guidance

由 Bayes 分解 $\nabla_{x_t}\log p(x_t\mid y_{\mathrm{obs}})=\nabla\log p(x_t)+\nabla\log p(y_{\mathrm{obs}}\mid x_t)$，用**同一去噪网络**一步重构 $\hat y=f_\theta(x_t,t)$ 来参数化引导分布，再经自动微分得到引导梯度：

1. **Mean Square (TSDiff-MS)**：$p(y_{\mathrm{obs}}\mid x_t)=\mathcal{N}(y_{\mathrm{obs}}\mid \hat y, I)$，等价观测段 MSE 引导。
2. **Quantile (TSDiff-Q)**：用非对称 Laplace（pinball）在多个分位数 $\kappa$ 上引导，更贴合 CRPS 评估。

引导强度 $s$ 在验证集选取；这是软约束，观测段不必严格硬拷贝。

### Prediction Refinement

详见 [[prediction-refinement|Prediction Refinement]]。

把扩散隐式密度当作先验能量，对基预测 $g(y_{\mathrm{obs}})$ 与观测拼成的 $\tilde y$ 做数据空间迭代：

- **Energy / LMC**：过阻尼 Langevin，$y\leftarrow y-\eta\nabla E+\sqrt{2\eta\gamma}\,\xi$。
- **Maximum Likelihood**：$\gamma=0$ 的梯度下降特例。
- 用单步 **representative diffusion step** $\tau$ 近似 $\log p_\theta(y)$ 的 ELBO，避免多步 $t$ 采样开销；默认 20 次精炼迭代。

### Linear Predictive Score (LPS)

定义 [[linear-predictive-score|LPS]] 为：在合成样本上拟合 ridge 回归后的**测试 CRPS**，作为廉价、可复现的 train-on-synthetic / test-on-real 指标。

## 关键结果

八个 GluonTS 单变量基准（Solar, Electricity, Traffic, Exchange, M4-Hourly, UberTLC, KDDCup, Wikipedia）：

- **Predict**：TSDiff-Q 在 5/8 数据集取得最低或次低 CRPS，与 [[csdi|CSDI]]、任务专用 TSDiff-Cond 及 DeepAR/TFT 等竞争；Q 引导整体优于 MS。
- **Missing context**：同一无条件模型在 RM / BM-B / BM-E（上下文 50% 缺失）上可与按场景训练的 TSDiff-Cond 竞争，无需为每种缺失重训。
- **Refine**：对 Seasonal Naive / Linear / DeepAR / Transformer，至少一种精炼设置在每个数据集降低 CRPS；点预测器上 LMC 更有利，概率基模型上 ML-Q 常更好。
- **Synthesize**：TSDiff 合成样本的 LPS 显著优于 TimeVAE / TimeGAN；用其训练 DeepAR/Transformer 在多数数据集也优于两基线，且接近用真实数据训练（真实数据还可用更长 lag/时间特征）。

## 贡献

1. 把无条件扩散 + 推理期 self-guidance 确立为时序概率预测的可行替代路线。
2. 提出 observation self-guidance（MS / Quantile），无需辅助分类器或 CFG 式联合训练。
3. 将扩散密度用作可插拔先验，对黑盒基预测器做后处理精炼。
4. 提出 LPS，并实证合成样本的下游预测质量。

## 局限性

- 完整 self-guidance 需迭代去噪，推理成本高于条件模型（Exchange 上约 201s vs Cond 163s）。
- 精炼质量依赖基预测器与代表步 $\tau$ 选择。
- 主实验为**单变量**；多元需额外跨特征层。
- 按数据集训练，非跨数据集 foundation 设定。

## 与相关源文件

- [[source-timegrad]]：条件 AR 扩散预测奠基；TSDiff 改为无条件 + 推理引导。
- [[source-csdi]]：条件扩散插补/预测；本文 Table 1 直接对比，并强调任务无关性。
- [[source-tsflow]]：同系作者线后续将无条件→条件桥接到 **Flow Matching + GP 先验**，并继续用 LPS 评生成质量。
