---
title: "Deep State Space Models for Time Series Forecasting"
type: source-summary
tags:
  - state-space-model
  - probabilistic-forecasting
  - time-series
  - kalman-filter
  - rnn
  - exogenous
  - neurips-2018
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Source: DeepState (Rangapuram et al., NeurIPS 2018)

**作者**: Syama Sundar Rangapuram, Matthias Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, Tim Januschowski (Amazon Research)  
**发表**: NeurIPS 2018  
**领域**: 概率时间序列预测；线性状态空间模型 × 深度 RNN

## 核心论点

DeepState 将经典线性[[kalman-filter|状态空间模型 (SSM)]]与深度循环网络结合：用**全局共享**的 RNN 从协变量映射出每条序列、每个时刻的线性 SSM 参数，再在 SSM 上用解析的 Kalman 滤波/平滑计算边际似然与预测后验。这样既保留 SSM 的可解释分量结构（水平/趋势/季节）与小样本数据效率，又能从大规模相关序列语料中联合学习共享模式。[^src-deepstate]

## 方法

### 问题设定

给定 \(N\) 条单变量序列 \(\{z^{(i)}_{1:T_i}\}\) 及时间协变量 \(\{x^{(i)}_{1:T_i+\tau}\}\)（预测区间协变量假定已知），目标是条件概率预测 \(p(z^{(i)}_{T_i+1:T_i+\tau}\mid z^{(i)}_{1:T_i}, x^{(i)}_{1:T_i+\tau};\Phi)\)。参数 \(\Phi\) 在所有序列间共享并联合学习；条件于协变量后假定序列独立，但通过共享 \(\Phi\) 传递统计强度。[^src-deepstate]

### 线性 SSM

潜状态 \(\ell_t\in\mathbb{R}^L\) 编码水平/趋势/季节，转移与观测为

\[
\ell_t = F_t \ell_{t-1} + g_t\varepsilon_t,\quad
z_t = a_t^\top \ell_t + b_t + \sigma_t\epsilon_t,
\]

\(\varepsilon_t,\epsilon_t\sim\mathcal{N}(0,1)\)，初值 \(\ell_0\sim\mathcal{N}(\mu_0,\mathrm{diag}(\sigma_0^2))\)。完整时变参数 \(\Theta_t=(\mu_0,\Sigma_0,F_t,g_t,a_t,b_t,\sigma_t)\)。经典做法对每条序列独立估 \(\Theta^{(i)}\)，无法跨序列共享，且协变量/结构选择成本高。[^src-deepstate]

### RNN 参数化映射

DeepState 用 LSTM 循环网络 \(h_t=h(h_{t-1},x_t,\Phi)\) 产生表示，再经仿射与逐元素变换映射到合法范围内的 \(\Theta_t=\Psi(x_{1:t},\Phi)\)。训练最大化训练窗边际似然

\[
\mathcal{L}(\Phi)=\sum_{i=1}^N\log p_{\mathrm{SS}}(z^{(i)}_{1:T_i}\mid\Theta^{(i)}_{1:T_i}),
\]

其中 \(p_{\mathrm{SS}}\) 由线性高斯 SSM 的 Kalman 滤波解析计算；在 MXNet 中用自动微分对 \(\Phi\) 做 SGD。预测时：在训练窗 unroll RNN 得 \(\Theta_{1:T}\)，Kalman 得 \(p(\ell_T\mid z_{1:T})\)；再在预测窗 unroll 得 \(\Theta_{T+1:T+\tau}\)，递推转移与观测采样 \(K\) 条轨迹。[^src-deepstate]

### 相对自回归模型的设计优势

与 DeepAR 等以目标值作输入的自回归 RNN 不同，DeepState **不把目标值直接喂入网络**，目标只通过似然进入训练。带来：(i) 对观测噪声更稳健；(ii) 缺失值可直接丢掉对应似然项；(iii) 预测期 RNN 只需 unroll 一次（与采样条数无关），而自回归模型需对每条样本路径重复 unroll。[^src-deepstate]

## 实验结果

- **参数可识别性（合成）**：按 day-of-week 季节 SSM 生成 5 组序列；随每组样本从 20→140，\(\mu_0\)、创新强度 \(\gamma_t\)、观测噪声 \(\sigma_t\) 逐步逼近真值，说明联合训练可恢复可解释 SSM 参数。[^src-deepstate]
- **小样本 electricity / traffic**：训练窗 2/3/4 周、预测 7 天；p50/p90 分位损失上 DeepState 多数设定优于 auto.arima、ets 与 DeepAR，尤其 2 周训练时显式季节结构带来优势。[^src-deepstate]
- **滚动日预测 vs MatFact**：无需按日重训，仅扩展训练窗并更新潜状态后验；p50 优于矩阵分解 MatFact，与 DeepAR 相当。[^src-deepstate]
- **公开语料**：M4-Hourly、tourism 月/季、parts 等上整体 p50/p90 最优或接近最优。[^src-deepstate]

## 贡献

1. 提出 **RNN→线性 SSM 参数** 的全局共享参数化，统一小样本结构先验与大规模联合学习。[^src-deepstate]
2. 保留可解释 SSM 分量与高效 Kalman 推断，同时用协变量编码促销等非线性外生效应。[^src-deepstate]
3. 非自回归目标输入带来缺失鲁棒与采样效率优势，并在多基准上优于经典 ETS/ARIMA 与同期 DeepAR。[^src-deepstate]

## 局限性

- 观测模型为**单变量高斯**；非高斯/计数需求依赖附录中的扩展方向，正文未系统验证。[^src-deepstate]
- 条件独立假设**不建模序列间相关**（仅通过共享 \(\Phi\) 迁移）。[^src-deepstate]
- 潜维 \(L\) 由季节粒度手工设定（如小时数据 hour-of-day + day-of-week → \(L=31\)），结构选择仍部分依赖领域知识。[^src-deepstate]
- 与后续扩散/流/[[k2vae|K²VAE]] 等灵活分布模型相比，表达力受线性高斯 SSM 族限制。[^src-deepstate]

## 关联页面

- [[deepstate]] — 方法实体
- [[deep-state-space-model]] — RNN 参数化线性 SSM 概念
- [[kalman-filter]] — 训练似然与预测后验的核心推断
- [[generative-time-series-forecasting]] — 概率预测谱系中的 SSM 分支
- [[k2vae]] — 后续 Koopman+Kalman 神经 SSM 路线
- [[timegrad]] — 后续扩散概率预测（对照）

[^src-deepstate]: [[source-deepstate]]
