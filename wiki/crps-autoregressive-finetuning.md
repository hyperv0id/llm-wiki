---
title: "CRPS Autoregressive Finetuning"
type: technique
tags:
  - crps
  - ensemble-calibration
  - finetuning
  - weather-forecasting
  - generative-models
  - probabilistic-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# CRPS Autoregressive Finetuning (CRPS 自回归微调)

**CRPS 自回归微调**是由 [[swift|Swift]] (arXiv 2025) 提出的技术：在多步自回归 rollout 上使用连续秩概率得分（CRPS）微调生成模型，使其产生校准良好的概率集合预报[^src-swift]。

## 问题背景

生成模型（扩散模型、一致性模型）通常使用 ℓ₂ 或类似点估计损失训练，这鼓励模型预测条件均值而非多样化样本。在天气预测等需要集合散布的应用中，这导致**预报散布不足**（SSR < 1）和**无法量化的预测不确定性**[^src-swift]。

传统解决方法是使用多模型集合（如 GenCast 依赖 4 个独立模型），增加了维护开销和扩展难度[^src-swift]。

## 技术细节

### CRPS 损失

CRPS 是一个严格的单变量预报评分规则，量化预报累积分布函数 (CDF) $F$ 与观测值 $y$ 之间的差异[^src-swift]：

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}[z \geq y])^2 \, \mathrm{d}z
$$

当预报分布由 $N$ 个有限集合成员表示时，公平无偏估计为[^src-swift]：

$$
\widehat{\text{CRPS}}(\hat{y}^{1:N}, y) = \frac{1}{N}\sum_n |\hat{y}^n - y| - \frac{1}{2N(N-1)}\sum_{n \neq n'} |\hat{y}^n - \hat{y}^{n'}|
$$

- **第一项**（MAE）：鼓励集合成员接近观测值
- **第二项**（自比较修正）：鼓励集合散布（成员间的多样性），防止所有成员坍缩到单一点估计

### 多步自回归反向传播

关键创新在于通过多个自回归步骤反向传播 CRPS 梯度[^src-swift]：

```
Input: x_0 (initial state), z ~ N(0, σ_d²I) (noise for N members)
For k = 1 to K:
    ŷ_k = f_θ(x_{k-1}, t=π/2) + x_{k-1}    # single-step NFE=1
    x_k = ŷ_k                                 # next initial condition
Loss = CRPS(ŷ_K, y_target)                    # computed on final step
Backprop through K steps
```

Swift 使用课程学习调度 $K = \{1,2,3,4,8\}$，每组 0.5–1.5M images。使用序列梯度检查点（sequential gradient checkpointing）节省内存[^src-swift]。

### 为什么对一致性模型可行

扩散模型每步需要 20–40 NFE，通过 K 步反向传播在时间和内存上**禁止性昂贵**。Swift 作为一致性模型，每步仅需 **NFE=1**，使多步微调变得可行[^src-swift]。

## 效果

| 指标 | 微调前 (Swift-B) | 微调后 (Swift) |
|------|-----------------|---------------|
| 长期稳定性 | < 15 天 | **75 天** |
| 短期 SSR (0–4 天) | ~1 | < 1 (恶化) |
| 长期 SSR (10+ 天) | << 1 | ~0.8 |

微调显著改善了长期预报稳定性和集合散布，代价是短期 SSR 的轻微退化（可由更好的学习率调度缓解）[^src-swift]。

## 局限

- **散布-技能权衡**：短期集合散布与长期稳定性之间存在零和博弈
- **计算成本**：尽管比扩散模型高效，多步微调仍然需要 5M images 额���训练
- **CRPS 平滑效应**：可能存在与 MSE 相似的过度平滑趋势[^src-swift]

## 相关页面

- [[swift]] — 首个应用 CRPS 自回归微调的模型
- [[autoregressive-consistency-models]] — 自回归一致性模型概念
- [[consistency-models]] — 一致性模型基础
- [[ensemble-forecasting-calibration]] — 集合预报校准

[^src-swift]: [[source-swift]]
