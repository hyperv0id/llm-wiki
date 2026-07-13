---
title: "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - autoregressive
  - ddpm
  - icml-2021
created: 2026-05-31
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Source: TimeGrad

**作者**: Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf (Zalando Research)
**发表**: ICML 2021
**领域**: 多变量概率时间序列预测

## 核心论点

TimeGrad 是首个将[[ddpm|DDPM]]扩散模型应用于多变量时间序列概率预测的工作[^src-timegrad]。核心洞察：将 RNN 用于"时间的自回归记忆编码"，将扩散模型用于"每个时间步内多变量联合分布的灵活建模"——RNN 把历史信息压缩为隐状态 $h_{t-1}$ 作为条件，扩散模型 $\varepsilon_\theta(x_t^n, h_{t-1}, n)$ 在该条件下学习任意复杂度的条件概率分布 $p(x_t^0 \mid h_{t-1})$[^src-timegrad]。这取代了 DeepAR 等方法的预设参数化输出分布（高斯、Copula），使模型能自然捕获跨维度的高阶非线性依赖和多模态性[^src-timegrad]。

## 方法

### 架构
- **时序编码器**：2 层 LSTM（隐维度 40），将历史观测 $x_{1:t-1}$ 和协变量 $c_t$ 编码为隐状态 $h_{t-1}$
- **条件扩散去噪网络 $\varepsilon_\theta$**：8 残差块膨胀卷积网络（WaveNet/DiffWave 风格），每块含 GAU 门控激活 $\sigma(\cdot) \odot \tanh(\cdot)$，残差通道=8，膨胀率交替 1 和 2[^src-timegrad]
- **条件注入**：$h_{t-1}$ 和扩散步 $n$ 的位置编码通过全连接层下/上采样后作为每层卷积偏置广播到 D 维

### 训练
完全继承 DDPM 的 $L_{\text{simple}}$ 范式：对预测窗口每个时间步 $t$，随机抽 $n \sim \text{Uniform}(1,...,N)$ 和 $\varepsilon \sim \mathcal{N}(0,I)$，最小化 $\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_n} x_t^0 + \sqrt{1-\bar\alpha_n} \varepsilon, h_{t-1}, n)\|^2$[^src-timegrad]。

### 推理
从白噪声出发，$N=100$ 步退火 Langevin 动力学去噪得到 $x_t^0$，送入 RNN 得 $h_t$，自回归重复至预测窗口结束。$S=100$ 条独立轨迹计算经验 CDF 用于 CRPS 评估[^src-timegrad]。

## 关键结果

在 6 个数据集上以 CRPS_sum 评估，TimeGrad 在 5 个（Solar, Electricity, Traffic, Taxi, Wikipedia）上排名第一，14 种基线中全面领先[^src-timegrad]。Transformer MAF（归一化流方法）是第二强方法，但在高维数据集上差距扩大——Wikipedia（D=2000）上 TimeGrad 0.0485 vs MAF 0.063，证明扩散模型在高维空间的表达优势随维度放大[^src-timegrad]。消融实验显示 $N \approx 10$ 即接近最优，$N=100$ 达最优，与图像扩散需要 $T=1000$ 形成鲜明对比——原因是 RNN 隐状态已提供了强引导信号，扩散仅需补充残差不确定性[^src-timegrad]。

## 贡献

1. 率先将扩散模型跨界引入时间序列概率预测领域[^src-timegrad]
2. 证明"概率预测不需要预设输出分布家族"——MSE 噪声回归可隐式学习任意复杂联合分布[^src-timegrad]
3. 证明 $N=100$ 步扩散对时序已足够（vs 图像的 1000 步）[^src-timegrad]
4. 为后续 [[csdi|CSDI]]、[[tsdiff|TSDiff]]（无条件 + self-guidance）、DiffSTG、SpecSTG 等扩散+时序工作奠定了范式基础或对照点[^src-timegrad]

## 局限性

- **推理速度慢**：预测 24 步需 $100 \times 24 = 2400$ 次 $\varepsilon_\theta$ 前向传播，实时场景不可行[^src-timegrad]
- **自回归串行依赖**：不同时间步必须串行，无法像 Transformer 并行预测[^src-timegrad]
- **固定 LSTM 瓶颈**：所有维度（D=8 到 D=2000）共用 h=40 的隐维度，高维信息压缩比不足[^src-timegrad]
- **统一噪声调度**：所有数据集使用相同 $\beta$ 线性调度，未针对数据特性调整[^src-timegrad]
- **缺少显式空间/拓扑归纳偏置**：与 GNN 方法不同，跨维度依赖完全依赖扩散模型容量隐式学习[^src-timegrad]

[^src-timegrad]: [[source-timegrad]]
