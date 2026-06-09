---
title: "K²VAE"
type: entity
tags:
  - time-series
  - probabilistic-forecasting
  - generative-model
  - vae
  - koopman-operator
  - kalman-filter
  - long-term-forecasting
  - one-step-generation
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# K²VAE

**K²VAE** (Koopman-Kalman Enhanced Variational AutoEncoder) 是一个面向**长期概率时间序列预测 (LPTSF)** 的 VAE 框架生成式模型，由华东师范大学 Wu 等人提出，发表于 ICML 2025（Spotlight）[^src-k2vae]。其核心思想是把概率预测重构为：在 **Koopman 测量函数空间**中对一个线性动力系统的**过程不确定性**进行建模与精炼，从而同时获得一步生成的高效率与长期预测的稳健性[^src-k2vae]。

K²VAE 是 [[generative-time-series-forecasting|生成式时间序列预测]] 中少见的 **VAE 路线 + 一步生成**代表，与扩散路线（[[timegrad|TimeGrad]]、[[csdi|CSDI]]）和流匹配路线（[[tsflow|TSFlow]]）形成对照。它与 [[d3vae|D³VAE]] 同属"VAE 一步生成"家族，但 D³VAE 面向短期预测，K²VAE 通过 [[koopman-linearization-for-forecasting|Koopman 线性化]] 与 Kalman 精炼专门解决长期场景[^src-k2vae]。

## 动机：长期概率预测的崩溃

作者实验显示，原生概率模型（GRU MAF、TimeGrad、CSDI）随预测步长延长，CRPS 急剧恶化，甚至劣于配高斯头的点预测模型（FITS、PatchTST、iTransformer）[^src-k2vae]。两个根因：

1. **非线性**：非平稳性与变量间复杂依赖使时间序列呈现非线性，难以写出简洁的状态转移方程，不确定性也难以量化[^src-k2vae]。
2. **误差累积 + 低效**：更长 horizon 带来更复杂的目标分布，扩散/流模型难找到清晰概率转移路径、每步迭代代价高，算力增加但性能反降[^src-k2vae]。

K²VAE 的回应是用 [[koopman-linearization-for-forecasting|Koopman 理论]] 把非线性系统线性化（处理根因 1），用 [[kalman-filter|Kalman 滤波]] 精炼并建模不确定性以缓解误差累积（处理根因 2），并用 VAE 一步生成替代多步采样（提升效率）[^src-k2vae]。

## 架构

K²VAE 由四个模块组成；KoopmanNet 与 KalmanNet 共同构成编码器[^src-k2vae]。

### 1. Input Token Embedding

将上下文序列 $X \in \mathbb{R}^{N\times T}$ 切成 $n$ 个非重叠 patch（patch size $s=T/n$），每个 patch 含全部 $N$ 个变量。**与通道独立 (Channel-Independent) 模型不同**，K²VAE 把多变量 patch 作为单个 token，flatten 后线性投影到 $d$ 维嵌入 $X^{P'}\in\mathbb{R}^{d\times n}$，从而在状态转移中隐式建模跨变量交互[^src-k2vae]。

### 2. KoopmanNet — 线性化

用可学习 MLP 作为测量函数 $\psi$，把 token 投影到测量空间 $X^{P*}$；在其上用 **one-step eDMD** 拟合局部 Koopman 算子：

$$K_{loc} = X^{P*}_{fore}(X^{P*}_{back})^{\dagger}$$

（$\dagger$ 为 Moore-Penrose 伪逆）。当 $\psi$ 欠拟合时，$K_{loc}$ 可能数值不稳定或把模型导向错误方向，故引入全局可学习部分 $K_{glo}$，最终 $K = K_{loc} + K_{glo}$，迭代外推得到重构上下文 $\hat{X}^C$ 与预测 horizon $\hat{X}^H$（在测量空间预测 $m=L/s$ 步）[^src-k2vae]。这一"有偏线性系统"是后续 KalmanNet 精炼的对象。

### 3. KalmanNet — 不确定性建模

详见 [[kalmannet-uncertainty-modeling]]。要点：先用 Encoder-Only Transformer **Integrator** 复用非线性残差 $X^{Res}=X^{P*}-\hat{X}^C$ 得到控制输入 $U$；构造 Process Model $z_k = Az_{k-1}+Bu_k+w_k$ 与 Observation Model $o_k = Hz_k+v_k$（把 $\hat{X}^H$ 作为先验观测），迭代 Predict/Update 步，用 Kalman 增益 $K_k$ 在观测与预测间加权精炼，输出精炼状态 $Z$ 与过程不确定性协方差 $P$[^src-k2vae]。所有矩阵 $A,B,H$ 可学习，过程/观测噪声协方差用下三角参数化 $Q=L_QL_Q^T,\,R=L_RL_R^T$ 保持正定[^src-k2vae]。跳跃连接 $Z' = Z + U$ 约束 Integrator 学习"残差的残差"[^src-k2vae]。

### 4. Decoder

变分分布 $Q(Z|X)=\mathcal{N}(Z', P)$，[[reparameterization-trick|重参数化]]采样后，由两个 MLP（逆测量函数 $\psi_\mu^{-1}, \psi_\sigma^{-1}$）映射回原空间，建模 $P(Y|Z)=\mathcal{N}(\mu,\sigma)$[^src-k2vae]。

## 学习目标

$$\mathcal{L} = \mathcal{L}_{ELBO} + \mathcal{L}_{Rec},\quad \mathcal{L}_{Rec}=\|X - X^{Rec}\|_2^2$$

其中 [[elbo|ELBO]] 保证 VAE 基本机制，先验 $P(Z|X)=\mathcal{N}(0,I)$ 期望测量空间线性系统收敛到稳定态；$\mathcal{L}_{Rec}$（把 Koopman 重构 $\hat{X}^C$ 映回原空间）促进测量空间的线性化[^src-k2vae]。

## 理论保证

- **定理 3.1（KalmanNet 稳定性）**：Update 步 $P_k=(I-K_kH_k)\hat{P}_k$ 在浮点运算下易失去正定。K²VAE 用对称化 $P_k=\frac12(P_k+P_k^T)$ 与 Joseph 形式 $P_k^{dual}=(I-K_kH_k)\hat{P}_k(I-K_kH_k)^T + K_kR_kK_k^T$ 保证正定[^src-k2vae]。
- **定理 3.2（K²VAE 收敛）**：当 $U\to 0$（线性系统偏差减小），KalmanNet 的状态转移方程收敛到 Koopman 算子；$A$ 可视为被 Kalman 增益增强、泛化更强的"微调 Koopman 算子"[^src-k2vae]。

## 实验结果

基于 **ProbTS** 基准，8 短期 + 9 长期数据集，对比 11 个基线（点预测：FITS / PatchTST / [[itransformer|iTransformer]] / Koopa；生成式：TSDiff / [[d3vae|D³VAE]] / GRU NVP / GRU MAF / Trans MAF / [[timegrad|TimeGrad]] / [[csdi|CSDI]]）；评测 CRPS 与 NMAE（5 次独立运行）[^src-k2vae]。

- **短期**：CRPS 降低 7.3%、NMAE 降低 14.5%（vs 次优 CSDI）[^src-k2vae]。
- **长期**：CRPS / NMAE 较 PatchTST 提升 20.9% / 19.9%；随 horizon (96/192/336/720) 延长保持优势，而多数基线显著退化[^src-k2vae]。
- **非平稳数据**：Exchange-S/L 上优势尤其明显——线性动力系统让不确定性更显式、更易建模[^src-k2vae]。
- **效率**：VAE 一步生成 + 轻量 MLP/线性层，达到最低显存与最快推理（Electricity-L 96-96 仅 0.094GB，远低于 CSDI 的 1.411GB）[^src-k2vae]。

### 消融

- **Koopman 算子**：混合 $K_{loc}+K_{glo}$ 优于单独任一；纯 $K_{loc}$ 在长期场景常因 eDMD 数值不稳定而失败[^src-k2vae]。
- **KalmanNet 连接**：完整（Integrator + 跳跃连接 + 控制输入）最优；去掉控制输入则 KalmanNet 缺乏精炼依据[^src-k2vae]。
- **模块**：KoopmanNet 与 KalmanNet 均不可或缺；因 KoopmanNet 负责线性化，对性能影响更大[^src-k2vae]。

## 局限性

- KalmanNet 基于**线性** Kalman 滤波，非线性建模能力弱[^src-k2vae]。
- one-step eDMD 对测量空间局部质量敏感，初始化病态时数值不稳定[^src-k2vae]。
- 仅单模态数值输入；零样本/基础模型方向尚未探索（作者列为未来工作）[^src-k2vae]。

## 关联页面

- [[koopman-linearization-for-forecasting]] — Koopman 线性化用于预测的通用范式
- [[kalmannet-uncertainty-modeling]] — K²VAE 的 KalmanNet 不确定性建模技术
- [[kalman-filter]] — Kalman 滤波基础
- [[variational-autoencoder]] — VAE 与 ELBO 框架
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[d3vae|D³VAE]] — 同属 VAE 一步生成家族（短期）
- [[timegrad|TimeGrad]]、[[csdi|CSDI]] — 扩散路线概率预测基线
- [[tsflow|TSFlow]] — 流匹配路线概率预测
- [[mmckm|MMCKM]] — 另一种 Koopman 时间序列模型（交通流，含控制输入）
- [[reparameterization-trick]]、[[elbo]] — VAE 数学基础

[^src-k2vae]: [[source-k2vae]]
