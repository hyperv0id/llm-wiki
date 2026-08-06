---
title: "Prediction Refinement"
type: technique
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - energy-based-model
  - langevin-dynamics
  - neurips-2023
created: 2026-07-13
last_updated: 2026-08-06
source_count: 2
confidence: medium
status: active
---

# Prediction Refinement

**Prediction Refinement** 是 [[tsdiff|TSDiff]]（NeurIPS 2023）提出的预测后处理技术：将无条件扩散模型学到的隐式概率密度用作**可插拔先验**，在**数据空间**（而非扩散潜空间）中迭代精炼任意基预测器的输出，无需改动训练流程或基预测器本身[^src-prs]。

## 动机

[[observation-self-guidance|Observation self-guidance]] 通过反向扩散实现条件化，每步需自微分求引导梯度，推理成本较高。Refinement 提供了一种**经济替代**：当基预测器已提供合理初值时，直接在数据空间中用少量迭代（默认 20 步）精炼，每次迭代只需对扩散模型做单步前向+反向——总开销低于完整 reverse diffusion[^src-prs]。此外，在工业场景中常面对黑盒生产预测系统，refinement 可以**后处理**方式提升精度而无需触及核心预测管线[^src-prs]。

## 能量函数形式

将精炼形式化为从正则化能量模型（EBM）中采样[^src-prs]：

$$
E_\theta(y; \tilde y) = -\log p_\theta(y) + \lambda R(y, \tilde y)
$$

- $y$：精炼目标（完整序列）
- $\tilde y$：观测段 $y_{\mathrm{obs}}$ 与基预测 $g(y_{\mathrm{obs}})$ 拼接成的初始序列
- $-\log p_\theta(y)$：扩散模型的负对数似然（先验），低能量 = 序列在扩散模型下合理
- $R(y, \tilde y)$：正则项，约束精炼结果不偏离初始预测太远
- $\lambda$：正则强度（论文固定为 1）[^src-prs]

低能量赋予同时满足「扩散模型合理」且「接近基预测」的序列[^src-prs]。

## 两种采样/优化方式

### 1. Energy-Based Sampling（LMC）

用**过阻尼 Langevin Monte Carlo** 采样[^src-prs]：

$$
y^{(i+1)} = y^{(i)} - \eta \nabla_{y^{(i)}} E_\theta(y^{(i)}; \tilde y) + \sqrt{2\eta\gamma}\,\xi_i,\quad \xi_i \sim \mathcal{N}(0, I)
$$

- $\eta$：步长
- $\gamma$：噪声尺度；$\gamma>0$ 时引入随机性，鼓励探索能量景观

### 2. Maximizing the Likelihood（ML）

$\gamma=0$ 的退化特例，等价于正则化梯度下降寻优[^src-prs]：

$$
\arg\min_y \left[-\log p_\theta(y) + \lambda R(y, \tilde y)\right]
$$

因目标非凸，收敛依赖初值 $y^{(0)} = \tilde y$——实验中基预测器质量显著影响 ML 精炼效果[^src-prs]。

### 正则项选择

与 observation self-guidance 类似，两种似然对应两种正则[^src-prs]：

| 变体 | $R$ | 含义 |
|------|-----|------|
| **LMC-MS / ML-MS** | MSE | 鼓励精炼序列与初值在欧氏距离上接近 |
| **LMC-Q / ML-Q** | Quantile loss (pinball) | 分位数对齐，更贴 CRPS 评估 |

## log p_θ(y) 的高效近似

完整 ELBO 近似需采样多个扩散步 $t$，推理代价高。TSDiff 提出用**代表扩散步（representative step）** $\tau$ 做单步近似[^src-prs]：

$$
\tau = \arg\min_{\tilde t} \mathbb{E}_{\epsilon,t,y}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right] - \mathbb{E}_{\epsilon,y}\left[\|\epsilon_\theta(x_{\tilde t}, \tilde t) - \epsilon\|^2\right]
$$

即选训练损失最接近**平均损失**的扩散步，使得单步近似既准确又高效[^src-prs]。$\tau$ 每数据集后训练计算一次，可复用。

## 实验结果

在 8 个 GluonTS 基准上，对 Seasonal Naive、Linear、DeepAR、Transformer 四种基预测器精炼[^src-prs]：

- **每个数据集**至少一种精炼设置降低了 CRPS[^src-prs]。
- **点预测器**（Seasonal Naive、Linear）：LMC 优于 ML，因加性噪声帮助探索能量景观、改善了初始点预测的「散布」[^src-prs]。
- **概率预测器**（DeepAR、Transformer）：ML-Q 常优于 LMC，表明概率基预测本身已提供足够多样性，无需额外采样噪声[^src-prs]。
- 弱基线上提升最显著（如 Seasonal Naive 在 Solar 上 CRPS 0.512→0.480），强基线上也有稳定收益[^src-prs]。

## 与 Observation Self-Guidance 的对比

| 维度 | Observation Self-Guidance | Prediction Refinement |
|------|--------------------------|----------------------|
| 空间 | 扩散潜空间（$x_t$） | 数据空间（$y$） |
| 起点 | 随机噪声 $x_T \sim \mathcal{N}(0,I)$ | 基预测器输出 + 观测拼接 |
| 是否需基预测器 | 否 | 是 |
| 推理开销 | 完整 reverse diffusion + 每步自微分 | 少量数据空间迭代 |
| 精度依赖 | 引导尺度 $s$ | 基预测器质量 + 代表步 $\tau$ |

## 同族方案：PIR 的识别–修订式后处理

**[[pir|PIR]]**（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）是另一类预测后处理技术：先用两层全连接网络估计逐实例的预测误差（以 MSE 为代理、MAE 辅助约束）识别可能的失效实例，再用局部修订（协变量预测与外生信息经 Transformer 注意力融合协变量与外生表示）与全局修订（训练集实例检索的 top-K 软加权求和）加权融合，直接产出修订后的点预测 [^src-pir]。与 TSDiff 精炼的对照：TSDiff 把扩散模型的隐式密度当作能量先验，在数据空间迭代优化基预测初值（概率式、不需要协变量与外生信息）；PIR 则是误差估计驱动的加权融合，需要可用的协变量/外生信息与历史检索库 [^src-pir]。

## 相关页面

- [[tsdiff]] — 承载精炼能力的无条件扩散模型
- [[observation-self-guidance]] — 推理期自引导预测（互补方案）
- [[source-prs]] — 原始论文
- [[energy-based-model]] — EBM 基础
- [[langevin-dynamics]] — LMC 采样基础
- [[linear-predictive-score]] — 同文提出的合成数据指标

[^src-prs]: [[source-prs]]
[^src-pir]: [[source-pir]]
