---
title: "Elucidating the SNR-t Bias of Diffusion Probabilistic Models"
type: source-summary
tags:
  - diffusion-model
  - snr-t-bias
  - wavelet-correction
  - cvpr-2026
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: high
status: active
---

# Elucidating the SNR-t Bias of Diffusion Probabilistic Models

> Yu, M., Sun, L., Zeng, J., Chu, X., & Zhan, K. (2026). Elucidating the SNR-t Bias of Diffusion Probabilistic Models. **CVPR 2026**. AMAP Alibaba Group & Lanzhou University.
> arXiv:2604.16044

## 核心贡献

本文发现并理论证明了扩散模型中的 **SNR-t Bias**（信噪比-时间步偏置）——逆过程推理中预测样本的 SNR 与指派时间步之间出现错配——并提出 **DCW（Differential Correction in Wavelet Domain）**，一种无需训练、即插即用的后处理方法，在小波域对不同频率子带分别做差分校正，以极低计算开销（0.08%~0.47%）显著提升生成质量。

## 关键发现

**Key Finding 1（滑动窗口实验）**：固定网络 $\epsilon_\theta(\cdot, s)$ 的时间步 $s$，输入具有不同 SNR 的样本 $x_t$（通过前向过程 Eq. 2 生成），网络输出 $\|\epsilon_\theta(x_t, s)\|_2$ 呈现系统性偏差：当输入 SNR 低于预期（$t > s$）时输出被高估；当 SNR 高于预期（$t < s$）时输出被低估。[^src-2604.16044]

**Key Finding 2（逆过程 SNR 始终偏低）**：在同一时间步 $t$，逆过程预测样本 $\hat{x}_t$ 的网络输出 $\|\epsilon_\theta(\hat{x}_t, t)\|_2$ 始终大于前向样本 $\|\epsilon_\theta(x_t, t)\|_2$，说明 $\hat{x}_t$ 的 SNR 系统性低于理想值。[^src-2604.16044]

## 数学形式化

### 前向过程

扩散模型前向过程定义为马尔可夫链：

$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I) \tag{Eq. 1}$$

一步加噪的闭式表达（重参数化）：

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I),
\quad \bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i) \tag{Eq. 2}$$

后验分布（贝叶斯定理）：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I) \tag{Eq. 3}$$

其中：

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t, \quad
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

### 逆过程与训练目标

神经网络 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t I)$ 近似后验。通过重参数化：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) \tag{Eq. 4}$$

重建样本（$x_0$ 的预测）：

$$x_\theta^0(x_t, t) = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \tag{Eq. 5}$$

训练目标：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon_t \sim \mathcal{N}(0,I)} \left[ \| \epsilon_\theta(x_t, t) - \epsilon_t \|_2^2 \right] \tag{Eq. 6}$$

### SNR 定义与错配

训练阶段，样本 SNR 严格耦合于时间步：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t} \tag{Eq. 7}$$

推理阶段，由于预测误差和数值求解器的离散化误差，逆过程偏离理想路径：

$$\hat{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \hat{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\hat{x}_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I) \tag{Eq. 8}$$

### 理论分析（核心）

**假设 5.1**：重建样本可表达为真实数据的线性组合加噪声：

$$x_\theta^0(\hat{x}_t, t) = \gamma_t x_0 + \phi_t \epsilon_t, \quad 0 < \gamma_t \leq 1, \; \phi_t < M \tag{Eq. 10}$$

**为什么之前的假设 $x_\theta^0 = x_0 + \phi_t \epsilon_t$ 是错误的？** 由于方差恒等式 $\mathbb{E}[\|x_0\|^2] = \|\bar{x}_0\|^2 + \text{Var}(\|x_0\|)$ 以及方差的非负性，可得 $\|\bar{x}_0\|^2 \leq \mathbb{E}[\|x_0\|^2]$。代入 $x_\theta^0$ 作为均值估计，得到：

$$\mathbb{E}[\|x_\theta^0\|^2] \leq \mathbb{E}[\|x_0\|^2] \tag{Eq. 11}$$

而 $x_\theta^0 = x_0 + \phi_t \epsilon_t$ 意味着 $\mathbb{E}[\|x_\theta^0\|^2] = \mathbb{E}[\|x_0\|^2] + \phi_t^2$，与 Eq. 11 矛盾。因此必须引入 $\gamma_t < 1$ 表示重建过程中的信息损失。[^src-2604.16044]

**定理 5.1**：逆过程中偏置去噪样本 $\hat{x}_t$ 的实际 SNR 为：

$$\text{SNR}(t) = \frac{\hat{\gamma}^2_t \bar{\alpha}_t}{1 - \bar{\alpha}_t + \left( \frac{\sqrt{\bar{\alpha}_t} \beta_{t+1}}{1-\bar{\alpha}_{t+1}} \phi_{t+1} \right)^2} \tag{Eq. 12}$$

其中 $0 < \hat{\gamma}_t \leq 1$。对比前向 SNR $\text{SNR}(t) = \bar{\alpha}_t/(1-\bar{\alpha}_t)$，逆过程的 SNR 严格更低（因为分母多了正项 $(\cdot)^2$），严格证明了 SNR-t bias 的存在。[^src-2604.16044]

**推导概览**：将 Eq. 10 和 Eq. 2 代入实际逆过程 Eq. 8，得到 $\hat{x}_{t-1}$ 的解析形式：

$$\hat{x}_{t-1} = \hat{\gamma}_{t-1} \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} + \left( \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \phi_t \right)^2} \, \epsilon_2 \tag{Eq. 14}$$

更简洁的形式（利用前向 Eq. 2 拼合）：

$$\hat{x}_{t-1} = \hat{\gamma}_{t-1} x_{t-1} + \psi_{t-1} \epsilon_3 \tag{Eq. 15}$$

## DCW 方法

### 像素空间差分校正

差分信号包含了将 $\hat{x}_{t-1}$ 拉向理想 $x_{t-1}$ 的梯度信息：

$$\hat{x}_{t-1} - x_\theta^0(\hat{x}_t, t) = \hat{\gamma}_{t-1} \left( x_{t-1} - \frac{\gamma_t}{\hat{\gamma}_{t-1}} x_0 \right) + \eta_t \epsilon_t \tag{Eq. 16}$$

将该方向信息整合到每一步去噪中：

$$\hat{x}'_{t-1} = \hat{x}_{t-1} + \lambda_t (\hat{x}_{t-1} - x_\theta^0(\hat{x}_t, t)) \tag{Eq. 17}$$

### 小波域差分校正（DCW）

对 $\hat{x}_{t-1}$ 和 $x_\theta^0(\hat{x}_t, t)$ 分别做 DWT 得到四个子带 $f \in \{ll, lh, hl, hh\}$。$ll$ 为低频（形状/轮廓），$lh/hl/hh$ 为高频（纹理/细节）。对每个子带分别校正：

$$\hat{x}^f_{t-1} = \hat{x}^f_{t-1} + \lambda^f_t (\hat{x}^f_{t-1} - x^f_\theta(\hat{x}_t, t)) \tag{Eq. 18}$$

### 动态权重调度

利用逆过程方差 $\sigma_t$ 作为去噪进度的自然指示器：

低频校正系数：
$$\lambda^l_t = \lambda_l \cdot \sigma_t \tag{Eq. 20}$$
高频校正系数：
$$\lambda^h_t = (1-\lambda_h) \cdot \sigma_t \tag{Eq. 21}$$

设计动机：去噪早期（$\sigma_t$ 大）侧重低频校正以重建轮廓，后期（$\sigma_t$ 小）侧重高频校正以细化纹理。$\lambda_l, \lambda_h$ 为标量超参数（通过两阶段网格搜索确定）。[^src-2604.16044]

## 实验结果

| 模型 | 数据集 | 步数 | 基线 FID | +DCW FID | 降幅 |
|------|--------|------|----------|----------|------|
| IDDPM | CIFAR-10 32×32 | 20 | 13.19 | 7.57 | 42.6% |
| IDDPM | CIFAR-10 32×32 | 50 | 5.55 | 4.16 | 25.0% |
| ADM | ImageNet 128 | 20 | 12.28 | 10.34 | 15.8% |
| ADM | ImageNet 128 | 50 | 5.18 | 4.52 | 12.7% |
| IDDPM | LSUN Bedroom 256 | 20 | 18.69 | 11.03 | 41.0% |
| EDM | CIFAR-10 | 13 NFE | 22.62 | 12.92 | 42.9% |

DCW 还可叠加在已有偏置校正方法（ADM-ES、DPM-FR）之上进一步改进。在 FLUX、Qwen-Image 等现代文生图模型上也有效。计算开销仅 0.08%~0.47%。[^src-2604.16044]

## 局限性

- 小波域差分校正是 heuristic 的后处理，缺乏更深刻的理论 insight
- 超参数 $\lambda_l, \lambda_h$ 需针对每个模型调优（但对变化不敏感）
- 与 exposure bias 校正方法同为 post-processing，但理论动机和实验验证远强于同类方法

## 与 Exposure Bias 的关系

本文清晰地界定了 SNR-t bias 与先前工作研究的 exposure bias 之间的区别与联系：[^src-2604.16044]

- **Exposure bias** 关注的是样本间的分布偏移（$x_t$ 与 $\hat{x}_t$ 之间的差异），是 inter-sample 层面的偏置
- **SNR-t bias** 关注的是样本 SNR 与时间步之间的错配，是 sample-timestep 层面的偏置
- SNR-t bias 是 exposure bias 的根本原因之一——SNR 错配导致网络预测误差，预测误差累积又进一步加剧 SNR 错配

[^src-2604.16044]: [[source-snr-t-bias]]
