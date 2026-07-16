---
title: "TSDiff"
type: entity
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - unconditional-generation
  - self-guidance
  - neurips-2023
created: 2026-07-13
last_updated: 2026-07-25
source_count: 2
confidence: medium
status: active
---

# TSDiff

**TSDiff** 是面向时间序列的**无条件**去噪扩散模型，由 Kollovieh, Ansari 等提出于 NeurIPS 2023（论文 *Predict, Refine, Synthesize*）[^src-prs]。与 [[timegrad|TimeGrad]]、[[csdi|CSDI]]、SSSD 等**任务专用条件**扩散不同，TSDiff 只学习完整序列边缘 $p_\theta(y)$，再在推理期用 [[observation-self-guidance|observation self-guidance]] 条件化到任意 $p(y_{\mathrm{ta}}\mid y_{\mathrm{obs}})$[^src-prs]。同一检查点服务三条路径：概率预测、基预测精炼、合成数据生成[^src-prs]。

## 问题设定

令 $y\in\mathbb{R}^L$，观测时间指标集 $\mathrm{obs}$ 与目标集 $\mathrm{ta}$ 互补。目标是建模 $p(y_{\mathrm{ta}}\mid y_{\mathrm{obs}})$，覆盖标准预测、上下文含缺失的预测等特例[^src-prs]。训练只拟合无条件 $p_\theta(y)$，条件化推迟到采样[^src-prs]。

## 架构

骨架来自 SSSD 对 DiffWave 的修改：残差块内 **S4** 处理时间维，**Conv1×1** 处理通道维[^src-prs]。

| 组件 | 设定 |
|------|------|
| 序列 | 单变量长度 $L$（小时数据约 360，日数据约 390） |
| 通道 | 原始序列 + lag 特征 → $x_t\in\mathbb{R}^{L\times C}$ |
| 残差 | 3 层 × 64 通道 |
| 扩散 | $T=100$，线性 $\beta_1=10^{-4}$，$\beta_T=0.1$ |
| 归一化 | GluonTS mean scaler（按上下文绝对值均值缩放） |
| 训练 | Adam lr=$10^{-3}$，1000 epoch，batch 64 |

输出维度与输入一致，符合无条件扩散惯例[^src-prs]。多元扩展可通过在 S4 后加跨特征层（如 Transformer）实现，但正文实验为单变量[^src-prs]。

条件对照 **TSDiff-Cond** 在残差块中以 Conv1×1 注入观测与掩码，结构接近 SSSD，用于消融“条件训练 vs 推理引导”[^src-prs]。

## 三条用例

### 1. Predict — Observation Self-Guidance

反向步在无条件均值上叠加引导梯度（详见 [[observation-self-guidance]]）[^src-prs]：

$$
p_\theta(x_{t-1}\mid x_t,y_{\mathrm{obs}})=\mathcal{N}\big(\mu_\theta(x_t,t)+s\sigma_t^2\nabla_{x_t}\log p_\theta(y_{\mathrm{obs}}\mid x_t),\,\sigma_t^2 I\big)
$$

- **TSDiff-MS**：高斯 / MSE 引导。
- **TSDiff-Q**：非对称 Laplace / 多分位 pinball 引导，通常 CRPS 更优[^src-prs]。

在 8 个 GluonTS 基准上，TSDiff-Q 在 5/8 取得最低或次低 CRPS，与 [[csdi|CSDI]]、DeepAR、TFT 及 TSDiff-Cond 同级，且**无需按任务重训**[^src-prs]。上下文 50% 缺失（RM / BM-B / BM-E）时，同一无条件模型仍可与按场景训练的 Cond 竞争[^src-prs]。

### 2. Refine — 能量先验精炼

将扩散密度当作先验能量 $E_\theta(y;\tilde y)=-\log p_\theta(y)+\lambda R(y,\tilde y)$，对基预测器 $g$ 输出迭代更新（LMC 或 $\gamma=0$ 的 ML）。详见 [[prediction-refinement|Prediction Refinement]][^src-prs]。用单步 representative $\tau$ 近似 ELBO，默认 20 次迭代，开销通常低于完整 reverse diffusion[^src-prs]。对 Seasonal Naive、Linear 等弱基线提升显著，对 DeepAR / Transformer 也常有收益[^src-prs]。

### 3. Synthesize — 下游训练

无条件采样生成合成序列；下游用 [[linear-predictive-score|LPS]]（合成数据上 ridge 的测试 CRPS）及 DeepAR/Transformer 评估[^src-prs]。TSDiff 样本在 LPS 上显著优于 TimeVAE / TimeGAN，多数设置也优于两基线训练的强预测器[^src-prs]。

## 与相关方法

| 方法 | 训练 | 条件化 | 任务重心 |
|------|------|--------|----------|
| [[timegrad\|TimeGrad]] | 条件 AR 扩散 | RNN 隐状态 | 多元逐步预测 |
| [[csdi\|CSDI]] | 条件扩散 + 掩码 SSL | 观测值硬条件 | 插补 / 预测 |
| SSSD | 条件 + S4 | 观测/掩码 | 插补 / 预测 |
| **TSDiff** | **无条件** | **推理 self-guidance** | 预测 + 精炼 + 生成 |
| [[tsflow\|TSFlow]] | 无条件/条件 CFM | CPS + 向量场引导 / GP 条件先验 | 预测 + 生成（后续） |

[[tsflow|TSFlow]]（ICLR 2025）由同系作者将"无条件训练 → 推理条件化"推进到 **Flow Matching + GP 先验**，并继续沿用 LPS；实验中 TSFlow-Cond 以更少 NFE 超越含 TSDiff 在内的扩散基线[^src-tsflow]。

## 局限性

1. Self-guidance 每步需求导，推理慢于专用条件模型[^src-prs]。
2. 引导为软约束，观测对齐依赖尺度 $s$[^src-prs]。
3. 精炼依赖基预测与代表步 $\tau$[^src-prs]。
4. 主结果单变量、按数据集训练[^src-prs]。

## 代码

开源：<https://github.com/amazon-science/unconditional-time-series-diffusion>

## 相关页面

- [[source-prs]] — 论文摘要
- [[observation-self-guidance]] — 观测自引导技术
- [[linear-predictive-score]] — LPS 指标
- [[prediction-refinement]] — 能量先验精炼技术
- [[source-timegrad]] / [[timegrad]] — 条件 AR 扩散预测
- [[source-csdi]] / [[csdi]] — 条件扩散插补/预测
- [[source-tsflow]] / [[tsflow]] — 后续 CFM + GP 先验
- [[classifier-guidance]] — 经典分类器引导（需辅助网络）
- [[classifier-free-guidance]] — CFG（联合训练条件/无条件）
- [[generative-time-series-forecasting]] — 生成式时序预测总览
- [[ddpm]] — DDPM 基础
- [[energy-based-model]] — 精炼所用 EBM 视角
- [[langevin-dynamics]] — LMC 精炼采样

[^src-prs]: [[source-prs]]
[^src-tsflow]: [[source-tsflow]]
