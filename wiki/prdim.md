---
title: "PRDIM"
type: entity
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - mnar
  - expectation-maximization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# PRDIM

**PRDIM**（Missing Pattern Recognized Diffusion Imputation Model）是 Sim et al.（KAIST + 首尔国立大学，arXiv 2605.25439，2026 预印本）提出的扩散插补框架，**专攻 [[missing-not-at-random|MNAR]]（非随机缺失）**——目前少有的显式建模缺失模式的扩散插补方法[^src-prdim]。核心是用一个**模式识别器**（pattern recognizer）在 EM 框架下近似缺失掩码分布 $p(M\mid X)$，并在扩散反向过程中引导生成。

## 动机

现实中缺失常依赖**未观测值本身**（如病情恶化导致测量缺失），即 MNAR——比 MCAR/MAR 更现实，但缺失过程**非可忽略**，必须显式建模 $p_\phi(M\mid X^{obs}, X^{mis})$[^src-prdim]。然而多数扩散插补（[[csdi|CSDI]] 及后继）在 MCAR 式人工掩码下训练评测 $p_\theta(X\mid X^{obs,A}, A)$。PRDIM 指出：若真实缺失掩码 $M$ 与人工掩码 $A$ 分布差异大，这类模型难以泛化到真实缺失；并实证**插补原始缺失远难于插补人工缺失**。

## 方法

联合建模 $p_\theta(X)\,p_\phi(M\mid X)$，通过 **EM** 最大化观测数据与掩码的联合似然 $p_{\theta,\phi}(X^{obs}, M)$（$X^{mis}$ 为隐变量）。详见 [[pattern-recognizer-guidance]]。

### 两个阶段

| 阶段 | 内容 |
|------|------|
| **Phase 1：扩散预训练 / 预插补** | 条件扩散骨干（Observed Reconstruction Task）；用 **adjacent target masking**——人工缺失放在原始缺失**邻近**（时序沿时间轴、图像取上下左右相邻像素），而非 CSDI 的 MCAR 随机掩码，使骨干对任意缺失模式鲁棒 |
| **Phase 2：EM 迭代** | **M 步**：独立训练扩散 $\theta$ 与模式识别器 $\phi$（BCE）；**E 步**：扩散在 $M, X^{obs}$ 条件下生成 $X^{mis}$，模式识别器提供引导 |

### 关键设计

- **模式识别器 $D_\phi$**：判别器（源自 GAIN / not-MIWAE）预测每分量观测概率，BCE 训练[^src-prdim]。
- **MNAR 下的 ELBO**（Proposition 3.1）：联合对数似然下界 = 扩散 VLB + $\mathbb{E}[\log p_\phi(M\mid X_0)]$ + 熵项，把模式识别器嵌入扩散目标。
- **Hard EM**（而非 DiffPuter 的 soft EM）：增强对 $X^{mis}$ 分布的探索；EM 单调性保证联合对数似然逐迭代不减（Corollary 3.2）[^src-prdim]。
- **模式识别器引导**（Proposition 3.3）：反向得分 = 扩散得分 − $\nabla_{X_t}\mathcal{L}_{PR}$，与 [[classifier-guidance|分类器引导]]同构；$\hat{X}_0$ 用 [[tweedies-formula|Tweedie 公式]]估计。早期可用随机识别器（中性引导）。

## 实验结果

跨**三种模态**在 MNAR 下评测，对比 10 个基线（Mean；判别式 TimesNet/TimeMixer++/BRITS/SAITS；生成式 GP-VAE/not-MIWAE；扩散 CSDI/MTSCI/cDiffPuter）[^src-prdim]：

- **时序**（ETT、STOCK、PEMS-Bay）：原始缺失上较最强扩散基线 cDiffPuter 改善（如 RMSE 1.209→1.057、MAE 0.782→0.663、MRE 46.19→39.16），**out-of-sample（未见缺失）增益最显著**。
- **图像**（FMNIST、CelebA-HQ）：恢复眼/鼻/嘴等语义结构，而 vanilla 扩散仅用全局均值色填充。
- **表格**（5 个 UCI）：超越 DiffPuter、MissForest、MICE、HyperImpute 等。

## 消融与发现

- 去掉模式识别器、或 hard EM 换成 soft EM（= DiffPuter）均显著掉点——**显式缺失建模 + 迭代 EM 缺一不可**[^src-prdim]。
- 人工缺失率 $M{-}A$（10/50/90%）影响有限。
- **在 MCAR 下 PRDIM 优势消失**（识别器只学到随机性）——证实其收益是 **MNAR 特有的**[^src-prdim]。

## 与相关方法的关系

- **vs [[csdi|CSDI]]**：CSDI 条件扩散 + MCAR 人工掩码、忽略缺失过程；PRDIM 加模式识别器显式建模 $p(M\mid X)$ 处理 MNAR。
- **vs DiffPuter**：DiffPuter 仅训练联合扩散 + soft EM；PRDIM 同时训练扩散 + 模式识别器，用 hard EM。
- **vs [[nuwats|NuwaTS]] / [[t1|T1]]**：后两者假设随机缺失（mask-invariant 表示 / mask-aware embedding），PRDIM 专门处理**非随机**缺失，是缺失机制谱系上互补的一极。

## 关联页面

- [[missing-not-at-random]] — MNAR 缺失机制与可忽略性
- [[pattern-recognizer-guidance]] — PRDIM 的模式识别器 + EM + 扩散引导机制
- [[classifier-guidance]] — 同构的扩散条件引导
- [[tweedies-formula]] — 后验均值估计依据
- [[csdi]] — 条件扩散插补（MCAR 假设，PRDIM 基线）
- [[diffusion-models]] — 扩散模型总览
- [[nuwats]] / [[t1]] — 假设随机缺失的插补方法（对照）

[^src-prdim]: [[source-prdim]]
