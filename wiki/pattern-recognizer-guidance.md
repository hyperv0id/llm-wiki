---
title: "Pattern Recognizer Guidance (模式识别器引导)"
type: technique
tags:
  - data-imputation
  - diffusion-models
  - mnar
  - guidance
  - expectation-maximization
created: 2026-06-08
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Pattern Recognizer Guidance（模式识别器引导）

**Pattern Recognizer Guidance** 是 [[prdim|PRDIM]] (arXiv 2026) 的核心机制：训练一个判别器 $D_\phi$（"模式识别器"）近似缺失掩码分布 $p(M\mid X)$，并在扩散反向过程中用其梯度引导去噪，使插补结果与估计的缺失模式一致——从而在 [[missing-not-at-random|MNAR]] 下显式建模非可忽略的缺失过程[^src-prdim]。

## 模式识别器 $D_\phi$

沿 GAIN / not-MIWAE 的判别器思路：$D_\phi(X)$ 预测每个分量被观测的概率 $[p(M_d=1\mid X)]\in\mathbb{R}^D$，用二元交叉熵训练[^src-prdim]：

$$\mathcal{L}_{PR}(M, X, D_\phi) = -M^\top \log D_\phi(X) - (1-M)^\top \log\big(1 - D_\phi(X)\big)$$

## EM 框架

由于 $X^{mis}$ 不可观测，最大化联合似然 $p_{\theta,\phi}(X^{obs}, M)$ 被转化为 **EM（Expectation-Maximization）**问题（$X^{mis}$ 为隐变量）[^src-prdim]：

- **M 步（Maximization）**：独立训练扩散模型 $\theta$（捕获联合分布 $p_\theta(X^{obs}, X^{mis})$，X₀-prediction 扩散损失）与模式识别器 $\phi$（BCE 损失 $\mathcal{L}_{PR}$）。
- **E 步（Expectation）**：扩散模型在 $M, X^{obs}$ 条件下生成 $X^{mis}$，同时模式识别器提供引导信号，把生成偏向与估计缺失模式一致的插补。

PRDIM 采用 **hard EM**（而非 [[diffputer|DiffPuter]] 的 soft EM）以增强对 $X^{mis}$ 分布的探索；交替 E/M 步**单调增加**观测数据与掩码的联合对数似然（Corollary 3.2 EM 单调性）[^src-prdim]。

## 引导推导（Proposition 3.3）

联合分布的得分可分解为扩散得分 + 模式识别器引导项[^src-prdim]：

$$\nabla_{X_t}\log p_{\theta,\phi}(X_t\mid X_0^{obs}, M) \simeq \underbrace{\nabla_{X_t}\log p_\theta(X_t\mid X_0^{obs})}_{\text{扩散得分}} \;-\; \underbrace{\nabla_{X_t}\mathcal{L}_{PR}(M, \hat{X}_0, D_\phi^*)}_{\text{模式识别器引导}}$$

这与 [[classifier-guidance|分类器引导]] 同构——只是"分类器"换成了预测缺失掩码的模式识别器，引导信号来自负缺失概率 $\mathcal{L}_{PR}$。其中 $\hat{X}_0$ 由 [[tweedies-formula|Tweedie 公式]]给出后验均值估计：

$$\hat{X}_0 = f_\theta(X_t, t; X_0^{obs})\odot(1-M) + X_0^{obs}\odot M$$

DDPM 反向步（Eq. 17）[^src-prdim]：

$$X_{t-1} = \sqrt{\bar\alpha_{t-1}}\Big(\hat{X}_0 - \tfrac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\nabla_{X_t}\mathcal{L}_{PR}(M,\hat{X}_0,D_\phi)\Big) + \sqrt{1-\bar\alpha_{t-1}}\,\varepsilon$$

> [!note] 早期迭代的中性引导
> EM 早期可用**随机初始化**的识别器作引导——此时它无判别力，引导退化为近零向量，对生成中性无害[^src-prdim]。

## 与相关机制的关系

- **vs [[classifier-guidance|分类器引导]]**：结构相同（得分 + 条件对数梯度），但条件 $y$ 换成缺失掩码 $M$，分类器换成模式识别器 $D_\phi$；目标是缺失模式一致性而非类别。
- **vs [[csdi|CSDI]] 的条件扩散**：CSDI 把观测值作为条件直接注入去噪网络但假设 MCAR；PRDIM 额外用判别器显式建模 $p(M\mid X)$ 以处理 MNAR。
- **Phase 1 的 adjacent target masking**：人工缺失放在原始缺失**邻近**位置（时序沿时间轴、图像取相邻像素），而非 CSDI 的 MCAR 随机掩码，使扩散骨干对任意缺失模式更鲁棒[^src-prdim]。

## 关联页面

- [[prdim]] — 提出此机制的 MNAR 扩散插补模型
- [[diffputer]] — soft EM 交替的出处（DiffPuter, ICLR 2025），本文 hard EM 的对照
- [[missing-not-at-random]] — MNAR 缺失机制与可忽略性
- [[classifier-guidance]] — 同构的扩散条件引导技术
- [[tweedies-formula]] — 后验均值估计 $\hat{X}_0$ 的依据
- [[csdi]] — 条件扩散插补（MCAR 假设的对照）

[^src-prdim]: [[source-prdim]]
