---
title: "布朗桥扩散 (Brownian Bridge Diffusion)"
type: technique
tags:
  - diffusion-models
  - diffusion-bridge
  - brownian-bridge
  - deterministic-generation
  - time-series-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# 布朗桥扩散 (Brownian Bridge Diffusion)

**布朗桥扩散**是一类把扩散过程的**两端都固定**的扩散桥（diffusion bridge）技术：不同于标准扩散把数据腐蚀到无信息的高斯噪声（仅起点是数据、终点是纯噪声），布朗桥让前向过程在一个**预先确定的终点**而非自由的高斯先验处收束，从而模拟两个给定状态之间的随机过程轨迹。[^src-s2dbm] 它源自布朗运动等经典随机过程，是带特定边界约束的条件扩散模型。[^src-s2dbm]

## 与标准扩散的对比

| | 标准条件扩散（DDPM 式） | 布朗桥扩散 |
|---|---|---|
| 终点 $y_T$ | 标准高斯噪声 $\mathcal{N}(0,I)$ | 确定性先验 $h$（桥的端点） |
| 逆过程起点 | 从纯噪声采样 | 直接赋值 $y_T=h$ |
| 历史信息作用 | 仅条件化逆过程 | 既是条件、又是桥的端点 |
| 时序结构 | 预测"从噪声生长"，结构弱 | 预测"从先验出发"，保留结构 |

标准扩散的预测起源于纯噪声、缺乏时序结构，历史数据仅作条件、改善有限；布朗桥通过钉住终点把先验直接注入生成轨迹的起点，减少逆向估计中的随机性。[^src-s2dbm]

## 数学形式（[[s2dbm|S²DBM]] 实例）

在 [[s2dbm|S²DBM]] 的统一框架 $y_t = \hat\alpha_t y_0 + \hat\beta_t \epsilon + \hat\gamma_t h$ 中，布朗桥由如下系数选择实现：约束 $\hat\alpha_t$ 非负、单调递减，满足边界 $\hat\alpha_0=0,\ \hat\alpha_T=1$，并取 $\hat\gamma_t=1-\hat\alpha_t$、$\hat\beta_t=\sqrt{2\hat\alpha_t(1-\hat\alpha_t)}$。[^src-s2dbm] 前向闭式为：[^src-s2dbm]

$$q(y_t\mid y_0,h)=\mathcal{N}\big(y_t;\ \hat\alpha_t y_0+(1-\hat\alpha_t)h,\ 2\hat\alpha_t(1-\hat\alpha_t)I\big)$$

- 当 $t=0$：均值 $=y_0$，方差 $=0$（钉在数据端）。
- 当 $t=T$：$\hat\alpha_T=1$ ⟹ 均值 $=y_0$？——注意此时方差 $2\hat\alpha_T(1-\hat\alpha_T)=0$ 且 $\hat\gamma_T=0$，终点收束到确定值；S²DBM 据此在逆过程中**直接令 $y_T=h$** 作为起点。[^src-s2dbm]
- 方差项 $2\hat\alpha_t(1-\hat\alpha_t)$ 在中间步达到最大、两端为零，正是布朗桥"两端收紧、中间最不确定"的标志性形状。[^src-s2dbm]

## 后验方差缩放与确定性采样

布朗桥逆过程的后验方差可参数化为可调形式（[[s2dbm|S²DBM]] 沿用 BBDM / I³SB）：[^src-s2dbm]

$$\hat\sigma_t^2 = s\cdot\frac{(1-\hat\alpha_{t-1})(\hat\alpha_{t-1}-\hat\alpha_t)}{1-\hat\alpha_t}$$

- $s=0$：$\hat\sigma_t^2=0$，采样**完全确定性**（无任何高斯噪声），逆向均值是 $y_t,\hat y,h$ 的线性组合，类似 [[probability-flow-ode|DDIM/ODE]] 采样——这是布朗桥用于点对点预测的关键，使生成稳定无振荡。[^src-s2dbm]
- $s\neq 0$（如 $s=1,2$）：恢复含噪概率采样，用于不确定性量化；$s=2$ 时其形式与 DDPM 的 $\tilde\beta_t$ 一致。[^src-s2dbm]

## 谱系与时序应用难点

布朗桥扩散属于"扩散桥"大家族，相关工作包括 DDBM（成对分布间随机插值）、I²SB（图像复原的非线性桥）、BBDM（图像到图像翻译的布朗桥）、GOUB（广义 OU 桥 + Doob h-变换）、Bridge-TTS（薛定谔桥语音合成）等；这些在图像复原中以退化图作为信息先验。[^src-s2dbm] 在时序中，TimeBridge 用扩散桥建模先验与数据分布间的转移，但其线性样条插值先验不适合预测任务。[^src-s2dbm]

> [!note] 时序预测建桥的核心难点
> 时序的回看窗与预测窗长度不同，历史序列无法像图像修复中的退化图那样直接提供结构化先验，因此**不能直接在历史 $x$ 与未来 $y$ 间建桥**。[[s2dbm|S²DBM]] 的解法是先用线性**先验预测器 $F(\cdot)$** 把历史序列映射为同形状的确定性表示 $h$，再以 $h$ 作为桥的终点。[^src-s2dbm]

## 关联页面

- [[s2dbm]] — S²DBM，将布朗桥扩散用于时序预测的模型
- [[diffusion-models]] — 扩散模型统一框架
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[probability-flow-ode]] — 确定性 ODE 采样（与 s=0 无噪声采样相呼应）
- [[timegrad]] — TimeGrad，标准条件扩散时序预测（布朗桥要改进的范式）
- [[csdi]] — CSDI，S²DBM 去噪网络架构来源

[^src-s2dbm]: [[source-s2dbm]]
