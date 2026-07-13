---
title: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
type: source-summary
tags:
  - probabilistic-forecasting
  - time-series
  - autoregressive
  - rnn
  - negative-binomial
  - amazon
  - arxiv-1704-04110
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Source: DeepAR (Salinas, Flunkert & Gasthaus, arXiv 2017/2019)

**作者**: David Salinas, Valentin Flunkert, Jan Gasthaus (Amazon Research)  
**版本**: arXiv:1704.04110v3 (2019-02-22)；初版 2017-04  
**领域**: 概率时间序列预测；全局共享自回归 RNN

## 核心论点

DeepAR 针对“数千至数百万条**相关**时间序列”的工业预测场景，用**一条全局共享**的 LSTM 自回归网络联合学习所有序列，输出每步似然参数，再以祖先采样生成 Monte Carlo 轨迹，从而得到校准的联合预测分布与任意子区间分位数。相对每序列独立拟合的 ARIMA/ETS/状态空间模型，它用跨序列共享减轻过拟合与手工特征工程，并在多条真实数据集上相对当时最强基线约 **15%** 准确率提升。[^src-deepar]

## 方法

### 问题设定

对序列 \(i\)，在 conditioning range \([1,t_0-1]\) 与已知协变量 \(x_{i,1:T}\) 下，建模预测区间条件分布

\[
P(z_{i,t_0:T}\mid z_{i,1:t_0-1}, x_{i,1:T}).
\]

模型分布分解为逐步似然因式 \(Q_\Theta=\prod_{t=t_0}^T \ell(z_{i,t}\mid\theta(h_{i,t},\Theta))\)，其中

\[
h_{i,t}=h(h_{i,t-1},z_{i,t-1},x_{i,t},\Theta)
\]

由多层 LSTM 实现：自回归（消费 \(z_{i,t-1}\)）且循环（反馈 \(h_{i,t-1}\)）。编码器与解码器共享同一架构与权重；训练时 \(t_0\) 可置 0，对整窗最大化对数似然。[^src-deepar]

### 似然

网络直接预测分布参数 \(\theta\)。正文实验用：

- **高斯**：\(\mu\) 为 \(h\) 的仿射，\(\sigma\) 经 softplus 保证正；
- **负二项**（正计数）：均值 \(\mu\) 与形状 \(\alpha\) 均 softplus，\(\mathrm{Var}[z]=\mu+\mu^2\alpha\)。

亦可换 beta、Bernoulli、混合等，只要可采样且 log-likelihood 可微。[^src-deepar]

### 尺度处理（power-law 销量）

Amazon 零售销量速度近似幂律，跨序列量级差数个数量级。DeepAR 两项处理：

1. **按序列尺度因子 \(\nu_i\)**（文中启发式：条件窗均值）缩放自回归输入；对负二项将 \(\mu\) 乘 \(\nu_i\)、\(\alpha\) 除 \(\sqrt{\nu_i}\)，使非线性层工作在可比动态范围；
2. **按 \(\nu_i\) 加权采样**训练窗，避免高速度序列被均匀采样欠拟合。[^src-deepar]

消融显示：负二项 + 缩放 + 加权采样（完整 DeepAR）优于同架构的 `rnn-gaussian` 与无缩放/均匀采样的 `rnn-negbin`。[^src-deepar]

### 训练与预测

- 滑动窗增广：固定总长 \(T\) 与条件/预测相对长度；预测窗必须有真值；允许 \(t=1\) 落在序列起始前并用 0 填充，以学习新品冷启动。
- 训练 teacher-forcing 用真值 \(z\) 计算 \(h\)；预测时对 \(\ell(\cdot\mid\theta)\) 采样 \(\tilde z\) 回馈。文中称 scheduled sampling 无明显收益。
- 预测：先在条件窗 unroll 得 \(h_{t_0-1}\)，再祖先采样多条轨迹；对轨迹在任意 \([L,L+S)\) 求和后取经验分位数，支持一致的多跨度 \(\rho\)-risk。[^src-deepar]

### 协变量

时间特征（age、hour/day/week/month）、类目/item 嵌入等；预测区间协变量须已知。特征标准化为零均值单位方差。[^src-deepar]

## 实验结果

数据集：`parts`、`electricity`、`traffic`、Amazon `ec-sub` / `ec`（至约 53 万条周销量）。

- **parts / ec / ec-sub**（\(\rho\)-risk，相对最强已发表基线）：DeepAR 整体显著优于 Croston、ETS、Snyder 负二项 AR、ISSM 及两 RNN 消融；计数数据上高斯似然明显更差。[^src-deepar]
- **electricity / traffic** 点预测 ND/RMSE 优于 MatFact。[^src-deepar]
- **定性**：学会不同速度/年龄商品的季节与 80% 区间；相对 ISSM 的线性不确定性增长，DeepAR 从数据学到 Q4 升高再回落的非线性不确定性；打乱轨迹时间相关会破坏长跨度校准，说明样本路径捕获了时间相关。[^src-deepar]
- 实现：MXNet + 单 GPU；`ec` 端到端 <10h；预测 200 条样本。[^src-deepar]

## 贡献

1. 将 **全局共享 AR-RNN + 参数化似然 + MC 轨迹** 系统化用于工业规模概率预测。[^src-deepar]
2. 针对幂律量级差异给出 **输入/似然参数缩放 + 速度加权采样**，并验证负二项对间歇/计数需求的必要性。[^src-deepar]
3. 跨序列共享支持**极少历史/新品**预测，且协变量驱动季节与组依赖、减少手工模型选择。[^src-deepar]

## 局限性

- 预设参数化似然（高斯/负二项）表达力有限，难刻画强多峰/高维跨变量复杂依赖；后续 [[timegrad|TimeGrad]] 等以扩散替换输出分布。[^src-deepar]
- 自回归 teacher-forcing 与采样闭环存在 train–test 落差；长 horizon 误差累积成为后续 [[ar-vs-nar-decoding|AR vs NAR]] 与 [[probts|ProbTS]] 讨论的核心。[^src-deepar]
- 正文主实验为**单变量**相关序列集合，非显式多变量联合或时空图结构。[^src-deepar]
- 缺失观测给出 principled 处理（用采样填入并剔除似然项），但未报告完整实验。[^src-deepar]

## 关联页面

- [[deepar]] — 方法实体
- [[generative-time-series-forecasting]] — 概率/生成式预测谱系中的 AR-RNN 分支
- [[ar-vs-nar-decoding]] — 自回归解码权衡
- [[deepstate]] — 同期 Amazon 非目标输入 SSM 对照
- [[timegrad]] — 以扩散替换 DeepAR 式参数化输出
- [[probts]] — 将 DeepAR 类 AR 概率模型纳入统一基准

[^src-deepar]: [[source-deepar]]
