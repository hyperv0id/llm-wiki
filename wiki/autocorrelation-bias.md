---
title: "Autocorrelation Bias (Time-Series Forecasting)"
type: concept
tags:
  - time-series-forecasting
  - learning-objective
  - bias
  - mse
created: 2026-07-13
last_updated: 2026-11-09
source_count: 3
confidence: high
status: active
---

# 自相关偏差（Autocorrelation Bias）

**定义**：MSE 作为标签序列条件负对数似然估计量时，与真实 NLL 之间的差距，来源是未来步之间的条件自相关被 MSE 忽略。[[distdf|DistDF]] 用 Theorem 3.1 给出形式化，并把这一差距命名为 autocorrelation bias[^src-distdf]。

## Theorem 3.1：条件高斯下的似然偏差

设 $y_{\mid x}\in\mathbb{R}^{T}$ 为给定历史 $x$ 的标签序列，$\hat{y}_{\mid x}\in\mathbb{R}^{T}$ 为预测序列，且 $y_{\mid x}\sim\mathcal{N}(\hat{y}_{\mid x},\Sigma_{\mid x})$（模型输出即条件均值）。从条件负对数似然 $-\log p_{y\mid x}$ 中去掉与 $\hat{y}_{\mid x}$ 无关的项（$T\log 2\pi$ 与 $\log\lvert\Sigma_{\mid x}\rvert$），得到可用于训练的二次型目标[^src-distdf]

$$
\mathrm{PNLL}=\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_{\Sigma_{\mid x}^{-1}}^2 .
$$

与 $\mathrm{MSE}=\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_2^2$ 相减即得[^src-distdf]：

$$
\mathrm{Bias}=\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_{\Sigma_{\mid x}^{-1}}^2-\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_2^2,\qquad
\lVert v\rVert_{\Sigma_{\mid x}^{-1}}^2=v^{\top}\Sigma_{\mid x}^{-1}v,
$$

该项当且仅当 $\Sigma_{\mid x}=I$ 时为零，即未来各步在给定 $x$ 后条件独立。这就是「MSE 隐含条件独立假设」的精确含义。Appendix A.1 的 Theorem A.1 记同一结论。

## 与前作推导口径的差别

Time-o1（论文称为该方向的先行工作）从 $y$ 的边缘高斯分布出发推导这一偏差。DistDF 的 Footnote 1 修正了推导口径：应当假设条件分布 $\mathbb{P}_{y\mid x}$ 为高斯，从条件对数似然而非边缘似然出发[^src-distdf]。

口径差别有实际后果：偏差由 $\Sigma_{\mid x}$（给定 $x$ 后的剩余相关）决定，因此仅让 $y$ 的无条件协方差对角化不足以消掉它（§3.1）[^src-distdf]。

## 为什么它真实存在：条件相关的测量

直接算未来步之间的 Pearson 相关会把共因结构算进来：$y_t\leftarrow x\rightarrow y_{t'}$ 使两个未来步因为共享历史而相关，即使给定 $x$ 后二者独立。论文因此用偏相关（partial correlation）：给定历史序列 $x$ 后计算未来步两两相关，实现方式与 MATLAB `partialcorr` 相同（Appendix C）[^src-distdf]。[[fredf|FreDF]] 在同一问题上用的是 DML（double machine learning），把历史当混淆变量估计 $y_t\to y_{t'}$ 的因果强度[^src-fredf]。

测量结果（Traffic，$T=192$）[^src-distdf]：

| 对象 | 偏相关矩阵的非对角结构 |
|---|---|
| 原始标签 | 逾 $50.3\%$ 的非对角元 $\lvert\rho\rvert>0.1$ |
| FreDF 的频率分量 | 非对角量级下降，仍留不可忽略的残余相关 |
| Time-o1 的主成分 | 同上，残余相关未消失 |

对应 Fig. 1(a)–(c)。FreDF 一侧的对偶证据是在 Weather（$T=192$）上用 DML 测得的 $37.5\%$ 非对角元超过 $0.3$，经 Fourier 变换后降到约 $3.6\%$ 超过 $0.1$——但那是分量间的边缘结构，与这里要求的条件结构不是同一个量[^src-fredf]。

## 为什么变换法消不掉它

Theorem 3.1 只要求 $\Sigma_{\mid x}$ 对角，而 Fourier 变换与 PCA 保证的是 $\Sigma$ 对角。FreDF 的 Theorem 3.3 与 Time-o1 的 Lemma 3.2 给出的是分量边缘去相关，条件是 $\Sigma$（无条件下协方差）而不是 $\Sigma_{\mid x}$；分量在给定 $x$ 后仍可能相关，偏差因此残留（§3.1、Footnote 2）[^src-distdf]。

反过来，若某个变换能在每个 $x$ 上把 $\Sigma_{\mid x}$ 变成对角阵，逐点 MSE 就等价于条件 NLL，偏差随之消失。论文的论证是 Fourier 与 PCA 做不到这一点，因此放弃继续寻找变换、改为对齐条件分布（§3.1）[^src-distdf]。

## 三条处理路线

| 路线 | 做法 | 与 $\Sigma_{\mid x}$ 的关系 |
|---|---|---|
| [[fredf\|FreDF]] | 正交频域分量上的模长似然 + 时域 MSE 混合；技术见 [[frequency-enhanced-direct-forecast]] | 只让边缘协方差对角化，$\Sigma_{\mid x}$ 仍有非对角元[^src-fredf] |
| [[distdf\|DistDF]] | 不做变换、不做似然因式分解，用联合分布 Wasserstein 差异上界直接对齐条件分布；技术见 [[joint-distribution-wasserstein-alignment]] | 绕开 $\Sigma_{\mid x}$ 的显式建模，改由 $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})=0\Rightarrow\mathbb{P}_{y\mid x}=\mathbb{P}_{\hat{y}\mid x}$ 保证[^src-distdf] |
| [[qdf\|QDF]] | 保留二次型似然 $L_\Sigma=\lVert Y-\hat{Y}\rVert_{\Sigma^{-1}}^2$，双层学习 PSD 矩阵 $\Sigma$；技术见 [[quadratic-form-weighted-objective]] | 用 $\Sigma^{-1}$ 的非对角元显式建模残余条件依赖，对角元承接 [[heterogeneous-task-weights\|异构任务权重]][^src-qdf] |

三条路线的取舍不同：FreDF 换坐标系但不换似然形式，DistDF 换掉似然估计本身，QDF 保留二次型但把固定的单位阵换成可学习矩阵。三者都不修改骨干结构；训练阶段的额外计算分别是一次 FFT（FreDF）、$(H+T)\times(H+T)$ 的协方差与矩阵平方根（DistDF）、双层元学习（QDF），推理阶段三者都只跑骨干[^src-fredf][^src-distdf][^src-qdf]。

## 相关页面

- [[label-autocorrelation]] — 偏差所依赖的统计现象
- [[direct-forecast]] — 偏差所在的预测范式
- [[distdf]]、[[joint-distribution-wasserstein-alignment]]、[[bures-wasserstein]]
- [[fredf]]、[[qdf]]、[[source-fredf]]、[[source-qdf]]、[[source-distdf]]

---

[^src-distdf]: [[source-distdf]]
[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
