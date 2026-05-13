---
title: "扩散模型"
type: concept
tags:
  - generative-model
  - diffusion
  - vae
  - score-based
created: 2026-04-28
last_updated: 2026-05-13
source_count: 15
confidence: high
status: active
---

# 扩散模型

**扩散模型**是一类生成模型，通过逐步向数据添加噪声（前向过程），然后学习逆转该过程（反向/去噪过程），从而从纯噪声中生成新样本。[^src-understanding-diffusion-models]

## 与变分自编码器的关系

扩散模型可以理解为一种特殊的**马尔可夫层次变分自编码器（Markovian HVAE）**。具体而言，变分扩散模型（VDM）是 Markovian HVAE 在以下三个限制条件下的特例：[^src-understanding-diffusion-models]

1. **潜变量维度等于数据维度**：所有隐变量 $z_t$ 与数据 $x$ 具有相同的维度，不进行维度压缩。
2. **编码器是固定的线性高斯变换**：前向过程不是学习得到的，而是预先定义的高斯转移核，没有可训练参数。
3. **最终潜变量是标准高斯分布**：$p(z_T) = \mathcal{N}(0, I)$，即经过足够多的扩散步后，数据分布被完全破坏为纯噪声。

## 前向过程

前向过程是一个固定的马尔可夫链，逐步向数据添加高斯噪声：[^src-understanding-diffusion-models]

$$
q(x_t | x_{t-1}) = \mathcal{N}\left(\sqrt{\alpha_t}\, x_{t-1},\; (1 - \alpha_t) I\right)
$$

其中 $\alpha_t \in (0, 1)$ 是噪声调度参数。通过[[reparameterization-trick|重参数化技巧]]，可以直接从 $x_0$ 一步采样到任意时间步 $t$——将随机采样 $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)$ 重排为确定性变换加固定噪声 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon$，使前向过程每步可导，这是扩散模型端到端训练的根本保证[^src-bluuuuue-reparameterization-trick]：

$$
q(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right), \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$x_T$ 近似服从标准高斯分布。[^src-understanding-diffusion-models]

## 反向过程

反向过程学习逆转前向加噪过程。模型 $p_\theta(x_{t-1} | x_t)$ 被参数化为高斯分布，其均值由神经网络预测：[^src-understanding-diffusion-models]

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right)
$$

训练目标是最小化反向过程与真实后验 $q(x_{t-1} | x_t, x_0)$ 之间的 KL 散度，该后验在给定 $x_0$ 的条件下具有闭式解。[^src-understanding-diffusion-models]

## 三种等价的训练目标

扩散模型的训练目标有三种等价的参数化形式，它们对应不同的预测任务：[^src-understanding-diffusion-models]

1. **预测 $x_0$**：直接训练网络从噪声图像 $x_t$ 中恢复原始数据 $x_0$。
2. **预测噪声 $\varepsilon$**：训练网络预测添加到 $x_t$ 中的噪声分量。这是 DDPM 中最常用的形式，对应的简化损失函数为：
   $$
   L_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]
   $$
3. **预测得分 $\nabla \log p(x_t)$**：训练网络估计对数概率密度的梯度。

这三种形式通过重参数化在数学上相互等价（给定一种预测可推导出另外两种），但 Li & He (2025) 指出，**在流形假设下，预测目标的根本性质不同**：x-prediction 的输出位于低维数据流形上（on-manifold），而 ε-和 v-prediction 的输出散布在整个高维空间中（off-manifold）。当 patch/输入维度接近或超过网络容量时，只有 x-prediction 能正常工作，ε-/v-prediction 会灾难性失败。[^src-2511-13720]

x-prediction 最早可追溯到 DDPM 的原始代码[^src-2511-13720]，但长期未被作为主要预测目标。Li & He 提出的 [[jit|JiT]]（Just image Transformers）系统论证了 x-prediction 在高维空间的必要性，使用标准 ViT + 大 patch + x-prediction 在像素空间实现了有竞争力的生成质量（JiT-G/16: FID 1.82 @ ImageNet 256）。详见 [[x-prediction]]。

## 与基于得分的模型的联系

扩散模型与**基于得分的生成模型**之间存在深刻联系。通过 Tweedie 公式，给定 $x_t$ 时 $x_0$ 的条件期望可以表示为：[^src-understanding-diffusion-models]

$$
\mathbb{E}[x_0 | x_t] = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

这表明预测噪声 $\varepsilon_\theta(x_t, t)$ 等价于估计得分函数 $\nabla_{x_t} \log p(x_t)$，两者之间仅差一个常数因子。因此，训练一个去噪扩散模型本质上等价于在多个噪声水平上进行得分匹配。[^src-understanding-diffusion-models]

## 采样过程

采样通过迭代去噪进行，从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步应用反向转移核：[^src-understanding-diffusion-models]

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

这一过程可以看作是**朗之万动力学**的离散化模拟——每一步都沿着得分函数的估计方向移动，同时注入适量噪声以保持采样多样性。[^src-understanding-diffusion-models]

## 条件生成

扩散模型支持两种主要的条件生成策略：[^src-understanding-diffusion-models]

- **分类器引导（Classifier Guidance）**：利用一个预训练的分类器 $p(y | x_t)$ 的梯度来引导采样过程，使生成结果符合特定类别 $y$。采样时，在得分估计中加入分类器的对数梯度项。
- **无分类器引导（Classifier-Free Guidance）**：同时训练条件模型 $\varepsilon_\theta(x_t, t, y)$ 和无条件模型 $\varepsilon_\theta(x_t, t, \emptyset)$，在采样时对两者进行插值：
  $$
  \tilde{\varepsilon}_\theta = \varepsilon_\theta(x_t, t, \emptyset) + w \left( \varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \emptyset) \right)
  $$
  其中 $w \geq 0$ 控制引导强度。无分类器引导是目前最广泛使用的条件生成方法。[^src-understanding-diffusion-models]

## 前向过程的归纳偏置塑造

标准扩散模型的前向过程使用各向同性白噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$，对所有频率成分均匀施加噪声：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}$$

Jiralerspong 等人（2025）提出[[frequency-based-noise-control|频域噪声控制]]方法，将白噪声替换为频域塑形噪声 $\boldsymbol{\epsilon}^{(w)}$：[^src-2502-10236]

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}^{(w)}$$

频域塑形流程为：

$$\boldsymbol{\epsilon} \xrightarrow{\mathcal{F}} \mathbf{N}_{\text{freq}} \xrightarrow{w(\mathbf{f})} \mathbf{N}_{\text{freq}}^{(w)} \xrightarrow{\mathcal{F}^{-1}} \boldsymbol{\epsilon}^{(w)}$$

其中 $\mathbf{N}_{\text{freq}} = \mathbf{N}_{\text{real}} + i\mathbf{N}_{\text{imag}}$ 为复高斯随机场，$w(\mathbf{f})$ 为频域加权函数。由于高斯分布的傅里叶变换仍是高斯，频域加权保持高斯假设不变。

核心洞见：**前向加噪过程中被抹除的信息，恰好是去噪模型有压力去学习的信息**。因此通过选择性强调/抑制/跳过特定频段的噪声，可以引导模型聚焦于数据分布的特定方面。实验在 5 个数据集（MNIST, CIFAR-10, DomainNet-Quickdraw, WikiArt, CelebA）中的 3 个上取得了 FID/KID 的显著改善，并展示了选择性忽略被噪声破坏频段的能力。[^src-2502-10236]

Falck 等人（Microsoft Research, 2025）从 SNR 角度给出另一种频域解释：若 $y_0=Fx_0$ 且 $C_i=\operatorname{Var}[(y_0)_i]$，则 DDPM 在频率 $i$ 的 SNR 为 $s_t^{\mathrm{DDPM}}(i)=\bar\alpha_t C_i/(1-\bar\alpha_t)$；自然图像等数据的 $C_i$ 随频率升高而快速下降，因此高频在前向过程中更早、更快被破坏。[^src-equal-snr] 他们提出 [[equal-snr|EqualSNR]]，令傅里叶噪声协方差满足 $\Sigma_{ii}=cC_i$，使所有频率在同一时间步拥有相同 SNR；该方法在标准图像 FID 上与 DDPM 大体持平，同时明显改善高频谱统计。[^src-equal-snr]

## 局限性

扩散模型存在以下主要局限：[^src-understanding-diffusion-models]

- **采样速度慢**：生成一个样本需要数十到数千步的迭代去噪过程，远慢于 GAN 或 VAE 的单步生成。加速方法包括知识蒸馏、快速 ODE 求解器和一致性模型。
- **无压缩的潜变量表示**：由于潜变量维度等于数据维度，扩散模型不提供有意义的低维潜空间，难以进行语义编辑或插值操作。

## 关键实现

- **[[ddpm|DDPM]]**：2020 年 NeurIPS 论文，首次证明扩散模型可生成高质量图像，建立了与得分匹配的等价性[^src-ddpm]
- **[[ncsn|NCSN]]**：NCSN，DDPM 的重要前身，使用退火朗之万动力学采样
- **[[score-based-sde|Score-Based SDE]]**：2021 年 ICLR 论文，用 SDE 统一了 NCSN (SMLD) 和 DDPM，引入 PC 采样和概率流 ODE[^src-sde]
- **[[dpm-solver|DPM-Solver]]**：2022 年 NeurIPS，专用快速 ODE 求解器，利用半线性结构在约 10 步内生成高质量样本[^src-dpm-solver]
- **[[consistency-models|Consistency Models]]**：2023 年 ICML，单步生成模型，通过学习 PF ODE 轨迹映射实现快速生成[^src-consistency-models]
- **[[flow-matching|Flow Matching]]**：2023 年 NeurIPS，统一框架，将扩散模型视为 Flow Matching 的特例，支持 OT 等更优概率路径[^src-flow-matching]
- **[[shortcut-models|Shortcut Models]]**：2025 年 arXiv，单阶段少步/单步生成模型，通过步长调节和自一致性实现高效推理[^src-shortcut-models]
- **[[edm|EDM]]**：2022 年 NeurIPS 论文，系统梳理扩散模型设计空间，提出 Heun 二阶采样器、预处理技术和对数正态噪声分布[^src-edm]
- **[[jit|JiT]]**：2025 年 arXiv (MIT)，提出 x-prediction 在高维空间的必要性，使用标准 ViT + 大 patch 在像素空间实现有竞争力的生成[^src-2511-13720]

## 相关概念

- [[reparameterization-trick]] — 重参数化技巧，前向加噪过程的可微性保证
- [[frequency-based-noise-control]] — 频域噪声控制，通过操控噪声频谱塑造归纳偏置
- [[equal-snr]] — 通过 $\Sigma_{ii}=cC_i$ 实现所有频率等 SNR 加噪
- [[frequency-hierarchy-in-diffusion]] — DDPM 的低频到高频隐式生成层级
- [[frequency-diffusion]] — 频域扩散，频域噪声控制的具体技术实现
- [[inductive-bias-shaping]] — 归纳偏置塑造，显式引导模型学习特定方面
- [[freqflow]] — FreqFlow，频率感知流匹配，通过双分支架构显式建模频率成分（2026）
- [[snr-t-bias]] — SNR-t Bias，扩散模型推理阶段的信噪比-时间步错配偏置（CVPR 2026）
- [[dcw]] — DCW，小波域差分校正方法，缓解 SNR-t Bias 的无需训练方法
- [[frequency-aware-conditioning]] — 频率感知条件化概念
- [[diffusion-models]] — A broader overview of diffusion models covering DDPM, SMLD, SDE unification, and applications
- [[x-prediction]] — x-prediction，扩散模型中直接预测干净数据的参数化方式
- [[jit|JiT]] — JiT (Just image Transformers)，基于 x-prediction 的像素空间扩散模型

## 引用

[^src-ddpm]: [[source-ddpm]]
[^src-sde]: [[source-sde]]
[^src-dpm-solver]: [[source-dpm-solver]]
[^src-consistency-models]: [[source-consistency-models]]
[^src-flow-matching]: [[source-flow-matching]]
[^src-shortcut-models]: [[source-shortcut-models]]
[^src-edm]: [[source-edm]]
[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
[^src-bluuuuue-reparameterization-trick]: [[source-bluuuuue-reparameterization-trick]]
[^src-2502-10236]: [[source-2502-10236]]
[^src-equal-snr]: [[source-equal-snr]]
[^src-2511-13720]: [[source-back-to-basics-let-denoising-generative-models-denoise]]
