---
title: "KalmanNet Uncertainty Modeling (K²VAE)"
type: technique
tags:
  - kalman-filter
  - uncertainty-quantification
  - probabilistic-forecasting
  - variational-inference
  - koopman-operator
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# KalmanNet Uncertainty Modeling (K²VAE)

**KalmanNet** 是 [[k2vae|K²VAE]] 中负责**建模并精炼过程不确定性**的数据驱动模块，将经典 [[kalman-filter|Kalman 滤波]] 的 Predict/Update 递归神经网络化，并把输出的协方差矩阵直接对齐为 VAE 的变分后验分布[^src-k2vae]。它解决的核心问题是：[[koopman-linearization-for-forecasting|KoopmanNet]] 用数据驱动方式拟合测量函数 $\psi$ 与 Koopman 算子 $K$，二者之间存在偏差（"有偏线性系统"），需要一个机制来精炼预测并量化随之产生的不确定性[^src-k2vae]。

## 设计动机

KoopmanNet 构造的线性系统是有偏的：生成的重构 $\hat{X}^C$ 与真实测量 $X^{P*}$ 之间存在残差。Kalman 滤波天生用于精炼此类有偏线性系统——通过融合"预测"与"观测"并提取 Kalman 增益来更新状态估计与不确定性[^src-k2vae]。K²VAE 因此把概率预测重述为：在测量空间中对一个线性动力系统迭代执行 Kalman Predict/Update，所得协方差即过程不确定性[^src-k2vae]。

## 三个组成部分

### 1. Integrator（复用非线性残差作为控制输入）

用 Encoder-Only Vanilla Transformer 处理 KoopmanNet 的非线性残差 $X^{Res} = X^{P*} - \hat{X}^C$，输出控制信号 $U=[u_1,\dots,u_m]\in\mathbb{R}^{d\times m}$[^src-k2vae]。其作用是把被 Koopman 线性化"丢弃"的非线性信息重新注入线性系统，帮助更快收敛、并隐式调整测量空间的拓扑结构使其趋于线性[^src-k2vae]。

### 2. Process Model 与 Observation Model

$$z_k = A z_{k-1} + B u_k + w_k,\qquad z_0 = x_n^{P*}$$
$$o_k = H z_k + v_k$$

其中 $A$（状态转移）、$B$（控制输入）、$H$（观测）均为 $d\times d$ 可学习矩阵；KoopmanNet 的预测 $\hat{X}^H$ 被当作 Update 步中的**先验观测**[^src-k2vae]。过程噪声 $w_k\sim\mathcal{N}(0,Q)$、观测噪声 $v_k\sim\mathcal{N}(0,R)$，协方差用下三角参数化 $Q=L_QL_Q^T,\ R=L_RL_R^T$ 以保持正定[^src-k2vae]。

### 3. Predict / Update 迭代

**Predict**：$\hat{z}_k = A z_{k-1} + B u_k$，$\hat{P}_k = A P_{k-1}A^T + Q$。

**Update**：$K_k = \hat{P}_k H^T(H\hat{P}_k H^T + R)^{-1}$，$z_k = \hat{z}_k + K_k(\hat{x}^H_k - H\hat{z}_k)$，$P_k = (I - K_kH)\hat{P}_k$[^src-k2vae]。

迭代后得到精炼预测 $Z=[z_1,\dots,z_m]$ 与各 token 协方差 $P=[P_1,\dots,P_m]$，再经跳跃连接 $Z'=Z+U$ 约束 Integrator 学习残差的残差[^src-k2vae]。

## 对齐变分后验

KalmanNet 的输出直接定义 VAE 的变分分布 $Q(Z|X)=\mathcal{N}(Z', P)$——这是 K²VAE 的关键设计：**用 Kalman 滤波得到的过程不确定性赋予 VAE 潜空间清晰语义**，而非用一个普通编码器盲目回归 $\mu,\sigma$[^src-k2vae]。

## 数值稳定性（定理 3.1）

数据驱动训练中浮点误差易使 $P_k=(I-K_kH_k)\hat{P}_k$ 失去正定。K²VAE 用两步修正[^src-k2vae]：

1. **对称化**：$P_k \leftarrow \frac12(P_k + P_k^T)$；
2. **Joseph 形式**：$P_k^{dual} = (I-K_kH_k)\hat{P}_k(I-K_kH_k)^T + K_kR_kK_k^T$，把更新式分解为两个正定项之和，论文证明其与原式数值等价但更稳健[^src-k2vae]。

## 与 Koopman 理论的一致性（定理 3.2）

随训练推进，$\mathcal{L}_{Rec}$ 减小使线性系统偏差降低，控制输入 $U\to 0$，KalmanNet 的状态转移方程收敛到 Koopman 算子，Integrator 逐步"退出"——保证 KalmanNet 不违反 Koopman 理论假设，$A$ 成为被 Kalman 增益增强、泛化更强的"微调 Koopman 算子"[^src-k2vae]。

## 与 MMCKM 控制输入的对比

[[mmckm|MMCKM]] 也在 Koopman 演化中引入控制输入（CrossAttention 注入宏观流影响 $z_{t+1}=K_zz_t+B_zu_t$），但目的不同：MMCKM 用控制输入耦合微观/宏观双尺度动力学并保证 ISS 稳定性；K²VAE 用控制输入复用非线性残差以精炼不确定性并定义变分后验[^src-k2vae]。两者都体现了"Koopman 线性系统 + 控制项"这一通用模式。

## 关联页面

- [[k2vae|K²VAE]] — 母模型
- [[kalman-filter]] — Kalman 滤波基础
- [[koopman-linearization-for-forecasting]] — KoopmanNet 构造的有偏线性系统是 KalmanNet 的精炼对象
- [[variational-autoencoder]]、[[reparameterization-trick]] — 变分后验与采样
- [[mmckm]] — 另一种 Koopman + 控制输入设计（交通流）

[^src-k2vae]: [[source-k2vae]]
