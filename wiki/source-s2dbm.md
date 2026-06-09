---
title: "S2DBM: Series-to-Series Diffusion Bridge Model"
type: source-summary
tags:
  - diffusion-models
  - time-series-forecasting
  - diffusion-bridge
  - brownian-bridge
  - point-forecasting
  - arxiv-2024
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# 源摘要：Series-to-Series Diffusion Bridge Model (S²DBM)

**元信息**：Hao Yang, Zhanbo Feng, Feng Zhou, Robert C Qiu, Zenan Ling（华中科技大学、上海交通大学、中国人民大学）。arXiv:2411.04491v1，2024 年 11 月。[^src-s2dbm]

## 核心问题与动机

扩散模型在时间序列预测中善于建模复杂分布，但其内在随机性损害了确定性（点对点）预测的精度——标准前向过程把未来序列逐步腐蚀成标准高斯噪声，预测因此"从纯噪声出发"，缺乏时序结构，历史数据仅作为条件提供有限改善，导致预测不稳定、生成低保真样本，点预测精度落后于 Autoformer、PatchTST、DLinear 等确定性模型。[^src-s2dbm]

## 统一框架（Theorem 1）

论文首先把非自回归扩散时序预测模型整合进一个统一框架，证明它们本质等价、仅在系数与网络架构上不同。前向过程被写成 $y_t = \hat\alpha_t y_0 + \hat\beta_t \epsilon + \hat\gamma_t h$，由四个时变系数 $\hat\alpha_t, \hat\beta_t, \hat\gamma_t, \hat\sigma_t^2$ 刻画，其中 $h=F(x)$ 是融入先验知识的条件表示。[[csdi|CSDI]]、SSSD、TimeDiff、TMDM 都是该框架的特例（见论文 Table 1）。[^src-s2dbm]

## 方法（S²DBM）

基于该框架，S²DBM 用[[brownian-bridge-diffusion|布朗桥]]把扩散过程的两端都钉住（pin down both ends），构造历史与预测序列之间的"数据到数据"桥，逆过程可直接从 $y_T=h$ 出发而无需采样高斯噪声。取 $\hat\alpha_0=0,\hat\alpha_T=1$、$\hat\gamma_t=1-\hat\alpha_t$、$\hat\beta_t=\sqrt{2\hat\alpha_t(1-\hat\alpha_t)}$，前向闭式为 $q(y_t|y_0,h)=\mathcal{N}(\hat\alpha_t y_0+(1-\hat\alpha_t)h,\,2\hat\alpha_t(1-\hat\alpha_t)I)$。通过把后验方差参数化为 $\hat\sigma_t^2 = s\cdot\frac{(1-\hat\alpha_{t-1})(\hat\alpha_{t-1}-\hat\alpha_t)}{1-\hat\alpha_t}$，令 $s=0$ 即得无任何高斯噪声的确定性生成模型（类 DDIM），用于点预测；$s=1$ 时退化为含噪概率采样。[^src-s2dbm]

两个轻量化设计支撑该方法：（1）**先验预测器 $F(\cdot)$** 与**条件编码器 $E(\cdot)$** 均采用单层线性模型（出于简洁、可解释、高效），$F(x)$ 把历史序列映射为确定性条件 $h$ 作为桥的终点；（2）**标签引导数据估计**：去噪网络直接预测干净数据 $y_0$ 而非噪声（预测噪声会引入更多振荡），并借鉴 Informer 的 label 策略，把历史尾段与未来序列沿时间维拼成 $y^*$ 一并重建。去噪网络沿用 CSDI 架构但移除其原有条件模块。[^src-s2dbm]

## 结果与局限

在 Weather、ILI、Exchange、ETTh1/h2/m1/m2 七个数据集上，点预测取得 56 个基准里 21 个第一、6 个第二，全面超越 CSDI、TMDM、TimeDiff 等扩散方法，并与 iTransformer、DLinear、RLinear 等 SOTA 持平或更优；概率预测以 CRPS/CRPS_sum 衡量与 CSDI、TMDM 竞争性相当。消融显示标签策略平均降低 21% MSE、16% MAE，布朗桥相对标准条件 DDPM 显著减少预测振荡。[^src-s2dbm] 局限：实验仅限中小规模标准基准（最大 21 通道），未在大规模/高维数据上验证；线性 $F,E$ 可能限制对强非线性历史依赖的建模；T=50 步采样仍慢于纯回归模型。

## 关联页面

- [[s2dbm]] — S²DBM 方法详解页
- [[brownian-bridge-diffusion]] — 布朗桥扩散桥技术
- [[diffusion-models]] — 扩散模型总览
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[timegrad]] — TimeGrad，扩散时序预测的奠基方法
- [[csdi]] — CSDI，S²DBM 去噪网络的架构来源与基线
- [[simdiff]] — SimDiff，另一确定性扩散点预测方法

[^src-s2dbm]: [[source-s2dbm]]
