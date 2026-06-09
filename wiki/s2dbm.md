---
title: "S2DBM"
type: entity
tags:
  - diffusion-models
  - time-series-forecasting
  - diffusion-bridge
  - brownian-bridge
  - point-forecasting
  - deterministic-generation
  - arxiv-2024
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# S²DBM (Series-to-Series Diffusion Bridge Model)

**S²DBM** 是一个基于[[brownian-bridge-diffusion|布朗桥扩散]]的时间序列预测模型，核心思想是用布朗桥把扩散过程的**两端都钉住**（pin down both ends），从而减少逆向估计中的随机性，专为点对点（确定性）预测设计（arXiv:2411.04491, 2024）。[^src-s2dbm] 它针对的痛点是：标准扩散时序模型从纯高斯噪声出发生成预测，缺乏时序结构、点预测精度落后于 Autoformer、PatchTST、DLinear 等确定性模型。[^src-s2dbm]

## 统一框架（Theorem 1）

S²DBM 的贡献之一是把非自回归扩散时序模型整合进一个统一框架，论证它们本质等价、仅在系数与架构上不同。前向过程写为：[^src-s2dbm]

$$y_t = \hat\alpha_t\, y_0 + \hat\beta_t\, \epsilon + \hat\gamma_t\, h,\qquad \epsilon\sim\mathcal{N}(0,I)$$

其中 $h=F(x)$ 是从历史数据 $x$ 提取的条件表示（先验知识），$\hat\alpha_t,\hat\beta_t,\hat\gamma_t,\hat\sigma_t^2$ 为时变系数。初始分布为 $p_\theta(y_T)=\mathcal{N}(\hat\gamma_T h,\hat\beta_T^2 I)$。[[csdi|CSDI]]、SSSD、TimeDiff、TMDM 都是该框架在不同系数选择下的特例：CSDI/SSSD/TimeDiff 取 $\gamma_t=0$（标准扩散），TMDM 取 $\gamma_t=\sqrt{1-\bar\alpha_t}$；CSDI/SSSD/TMDM 预测噪声 $\epsilon$，TimeDiff 直接预测数据 $y_0$。[^src-s2dbm]

## 布朗桥实例化（Corollary 1）

S²DBM 通过特定系数选择把上述框架实例化为[[brownian-bridge-diffusion|布朗桥]]：约束 $\hat\alpha_t$ 非负、随 $t$ 单调递减，满足边界 $\hat\alpha_0=0,\hat\alpha_T=1$，并定义 $\hat\gamma_t=1-\hat\alpha_t$、$\hat\beta_t=\sqrt{2\hat\alpha_t(1-\hat\alpha_t)}$。[^src-s2dbm] 前向过程闭式为：[^src-s2dbm]

$$q(y_t|y_0,h)=\mathcal{N}\big(y_t;\ \hat\alpha_t y_0+(1-\hat\alpha_t)h,\ 2\hat\alpha_t(1-\hat\alpha_t)I\big)$$

由于 $\hat\alpha_T=1,\hat\gamma_T=0$，终点 $y_T$ 退化为 $h$（方差也趋于 0），逆过程可**直接赋值 $y_T=h$**，无需从噪声高斯先验采样——这正是"钉住两端"的含义，使模型捕获更多目标序列的结构信息。[^src-s2dbm]

逆过程转移为 $p_\theta(y_{t-1}|y_t,x)=\mathcal{N}(\kappa_t y_t+\lambda_t y_\theta+\zeta_t h,\hat\sigma_t^2 I)$，其中 $\kappa_t,\lambda_t,\zeta_t$ 由 $\hat\alpha_t$ 与 $\hat\sigma_t^2$ 解析确定。[^src-s2dbm]

## 确定性 vs 概率：方差缩放开关

S²DBM 把后验方差参数化为一个可调形式（沿用 [[brownian-bridge-diffusion|BBDM]] / I³SB 的思路）：[^src-s2dbm]

$$\hat\sigma_t^2 = s\cdot\frac{(1-\hat\alpha_{t-1})(\hat\alpha_{t-1}-\hat\alpha_t)}{1-\hat\alpha_t}$$

超参 $s$ 是同一训练模型下切换两种推理模式的开关（训练过程完全相同）：[^src-s2dbm]

- **点预测（Example 1）**：$\hat\alpha_t=1-\tfrac{t}{T}$、$s=0$ ⟹ $\hat\sigma_t^2=0$，采样完全**确定性**（类 DDIM），逆向均值是 $y_t,\hat y,h$ 的线性组合，**无任何高斯噪声**，保证稳定且精确的点对点预测。[^src-s2dbm]
- **概率预测（Example 2）**：$s=1$（或 $s=2$ 时形式与 DDPM 的 $\tilde\beta_t$ 一致），后验含噪，逆过程加入 $z\sim\mathcal{N}(0,I)$，用 100 个样本近似分布。[^src-s2dbm]

> [!note] 与确定性采样的联系
> $s=0$ 时 S²DBM 的逆过程与 [[probability-flow-ode|确定性 ODE 采样]] 精神一致——把扩散模型变成无噪声的生成器，这是它在点预测上超越含噪扩散基线的关键。[^src-s2dbm]

## 关键组件

### 线性先验预测器 $F(\cdot)$ 与条件编码器 $E(\cdot)$
时序预测中回看窗与预测窗长度不同，历史序列无法像图像修复中的退化图那样直接作为结构化先验，因此**不能直接在 $x$ 与 $y$ 间建桥**。S²DBM 用先验预测器 $F(\cdot)$ 把历史序列转成确定性条件表示 $h$（桥的终点 + 逆过程起点的引导），并用条件编码器 $E(\cdot)$ 产生引导逆过程的 $c=E(x)$。两者均为**单层线性模型**，理由是简洁、可解释、高效。[^src-s2dbm]

### 标签引导数据估计（Label-Guided）
去噪网络 $y_\theta$ **直接预测干净数据 $y_0$ 而非噪声**——作者发现预测噪声会在结果中引入更多振荡。借鉴 [[informer|Informer]] 的 label 策略，把历史尾段（label 长度）与未来序列沿时间维拼成 $y^*$，让网络同时重建已知段并预测未来段，更好捕获底层模式。消融显示该策略平均降低 21% MSE、16% MAE。[^src-s2dbm]

### 去噪网络架构
沿用 [[csdi|CSDI]] 的去噪网络架构，但**移除其原有条件机制相关模块**（条件改由独立的 $E,F$ 提供）。训练用 MAE 损失、Adam、T=50 扩散步、4 个残差层、8 残差通道。[^src-s2dbm]

## 性能

在 Weather、ILI、Exchange、ETTh1/h2/m1/m2 七个数据集（输入长 H=336）上：[^src-s2dbm]

- **点预测**：56 个基准中取得 **21 个第一、6 个第二**，全面超越 CSDI、TMDM、TimeDiff 等扩散方法，并与 iTransformer、DLinear、NLinear、RLinear 等确定性 SOTA 持平或更优。[^src-s2dbm]
- **概率预测**：CRPS / CRPS_sum 与 CSDI、TMDM 竞争性相当。[^src-s2dbm]
- **消融**：相比标准条件 DDPM（cDDPM），布朗桥显著减少预测振荡；替换为 CSDI 的 $E$ 或 $\mu_\theta$ 均明显变差，验证线性条件法与去噪架构的价值。[^src-s2dbm]

## 局限性

1. 实验仅限中小规模标准基准（最大 21 通道），未在大规模/高维场景验证。
2. 线性 $F,E$ 简洁高效但可能限制对强非线性历史依赖的表达力。
3. T=50 步迭代采样仍慢于一次前向的纯回归模型（如 DLinear）。[^src-s2dbm]

## 关联页面

- [[brownian-bridge-diffusion]] — 布朗桥扩散桥技术（S²DBM 的核心机制）
- [[diffusion-models]] — 扩散模型统一框架与应用总览
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[timegrad]] — TimeGrad，扩散时序预测奠基方法（S²DBM 框架收纳的前身之一）
- [[csdi]] — CSDI，S²DBM 去噪网络的架构来源与主要基线
- [[simdiff]] — SimDiff，另一条确定性扩散点预测路线（端到端 + MoM 集成）
- [[tedm]] — TEDM，O(H) 采样的 EDM 时序扩散
- [[nsdiff]] — NsDiff，非平稳扩散概率预测
- [[informer]] — Informer，label 策略来源
- [[probability-flow-ode]] — 确定性 ODE 采样（与 s=0 无噪声采样相呼应）

[^src-s2dbm]: [[source-s2dbm]]
