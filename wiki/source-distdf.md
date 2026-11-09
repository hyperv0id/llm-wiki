---
title: "DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment"
type: source-summary
tags:
  - time-series-forecasting
  - direct-forecast
  - learning-objective
  - optimal-transport
  - wasserstein
  - iclr-2026
created: 2026-07-13
last_updated: 2026-11-09
source_count: 1
confidence: high
status: active
---

# DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment

Hao Wang、Licheng Pan、Yuan Lu、Zhixuan Chu、Xiaoxi Li、Shuting He、Zhichao Chen、Qingsong Wen、Haoxuan Li、Zhouchen Lin（小红书、浙江大学、上海财经大学、北京大学、Squirrel AI；Haoxuan Li 与 Zhouchen Lin 为通讯作者），ICLR 2026，arXiv:2510.24574v3（PDF 首页标注 2026-10-06）。本地 `raw/2510.24574-distdf.pdf` / `.txt`，代码 `github.com/Master-PLC/DistDF`[^src-distdf]。

## 问题与机制

[[direct-forecast|DF]] 用 MSE 估计条件负对数似然，等价于假设未来各步条件独立。Theorem 3.1 把 MSE 与条件 NLL 之差写成 $\mathrm{Bias}=\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_{\Sigma_{\mid x}^{-1}}^2-\lVert y_{\mid x}-\hat{y}_{\mid x}\rVert_2^2$，仅当 $\Sigma_{\mid x}=I$ 时为零。前作 FreDF（Fourier）与 Time-o1（PCA）只保证分量的边缘去相关，不保证给定 $x$ 后的条件去相关；Traffic（$T=192$）偏相关矩阵逾 $50.3\%$ 非对角元 $|\rho|>0.1$，两种变换后仍留残余相关（§3.1、Fig. 1、Appendix C）[^src-distdf]。

DistDF 改为直接对齐 $\mathbb{P}_{\hat{y}\mid x}$ 与 $\mathbb{P}_{y\mid x}$。给定 $x$ 通常只有一条 $y$，条件差异无法估计（Footnote 3 指出这是所有分布差异度量的共同限制），于是引入联合分布 Wasserstein 差异 $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})$：Lemma 3.3 证明它是期望条件差异的上界，Theorem 3.4 证明其为零时几乎处处 $\mathbb{P}_{y\mid x}=\mathbb{P}_{\hat{y}\mid x}$。高斯假设下取 Bures–Wasserstein 闭式解（Lemma 3.5），与 MSE 混合为 $\mathcal{L}_{\mathrm{DistDF}}=\gamma\mathcal{L}_{\mathrm{Dist}}+(1-\gamma)\mathcal{L}_{\mathrm{MSE}}$（Eq. 6）[^src-distdf]。

## 证据

主实验 4 数据集（ETTm1、ETTh1、ECL、Weather），$H=96$，$T\in\{96,192,336,720\}$，骨干 TimeBridge 与 Fredformer。Table 1 的 8 组 4-horizon 均值中 DistDF 的 MSE 全部最低（TimeBridge + ETTm1 与 Time-o1 同为 $0.383$）。Table 2 的仅均值、仅协方差两条支路单独接入均降低 MSE，完整 DistDF 在四个数据集上 MSE 与 MAE 均最低；Table 3 在 EMD、MMD@Linear、MMD@RBF、KL、DF 对照中 16 组里 14 组最优；Table 4–5 的最优 $\gamma$ 落在 $0.001\text{--}0.1$，$\gamma=1$ 退化。附录覆盖经典模型横比（Table 6，40 项 MSE 第一）、5 种骨干、自回归滚动、D3U 概率预测、多尺度架构、变历史长度与 5 种子（Table 8–13）。训练额外开销低于 $1$ ms，推理零开销（Appendix D.5）[^src-distdf]。

## 局限

Bures–Wasserstein 在高斯假设下只匹配一阶与二阶矩，非高斯数据需高阶统计量才能完整刻画条件分布（Conclusion）；接入新数据集须联合调节学习率 $\eta$ 与权重 $\gamma\in[0.001,0.1]$（§4.1、Table 4–5）[^src-distdf]。同组其他路线见 [[fredf]]、[[source-fredf|FreDF 原论文]]、[[qdf]]、[[source-qdf|QDF 原论文]]。

机制与实现见 [[distdf]]；技术页面 [[joint-distribution-wasserstein-alignment]]、[[bures-wasserstein]]、[[autocorrelation-bias]]、[[label-autocorrelation]]、[[direct-forecast]]、[[optimal-transport]]。

[^src-distdf]: [[source-distdf]]
