---
title: "Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)"
type: source-summary
tags:
  - diffusion
  - guidance
  - conditional-generation
  - generative-models
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: high
status: active
---

# Classifier-Free Diffusion Guidance

Jonathan Ho & Tim Salimans, Google Research Brain team. NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598, July 2022.

## 核心贡献

提出了**无分类器引导**（Classifier-Free Guidance, CFG），一种不需要额外分类器的扩散模型条件生成方法。联合训练条件扩散模型和无条件扩散模型（通过随机丢弃条件信息），采样时对条件得分和无条件得分进行线性组合，实现与分类器引导相同的 FID/IS 权衡效果。[^src-classifier-free-diffusion-guidance]

## 动机

分类器引导（Dhariwal & Nichol, 2021）存在三个问题：
1. **额外训练成本**：需要训练一个能处理噪声输入的独立分类器，无法复用预训练分类器；
2. **对抗性疑虑**：分类器引导等价于用梯度对抗攻击欺骗分类器，可能只是对抗性地提高基于分类器的指标（FID、IS）而非真正提高生成质量；
3. **与 GAN 训练的相似性**：沿着分类器梯度方向更新生成器，类似于 GAN 的训练方式，可能只是继承了 GAN 对分类器指标的优势。[^src-classifier-free-diffusion-guidance]

## 方法

### 训练

单一神经网络同时参数化条件模型 $\epsilon_\theta(z_\lambda, c)$ 和无条件模型 $\epsilon_\theta(z_\lambda, c=\varnothing)$。训练时以概率 $p_\text{uncond}$ 将条件 $c$ 替换为空标记 $\varnothing$，使模型在有无条件两种模式下工作。[^src-classifier-free-diffusion-guidance]

### 采样

使用修正得分进行采样：

$$\tilde{\epsilon}_\theta(z_\lambda, c) = (1 + w)\,\epsilon_\theta(z_\lambda, c) - w\,\epsilon_\theta(z_\lambda)$$

其中 $w$ 是引导强度。$w=0$ 退化为标准条件生成；$w>0$ 放大条件影响，降低多样性同时提高样本保真度。[^src-classifier-free-diffusion-guidance]

### 隐式分类器解释

CFG 受隐式分类器 $p^i(c|z_\lambda) \propto p(z_\lambda|c)/p(z_\lambda)$ 的启发。若得分估计是精确的保守向量场，则 $\epsilon^*(z_\lambda, c) - \epsilon^*(z_\lambda)$ 正比于该隐式分类器的梯度。但实际中神经网络产生的得分估计不一定是保守向量场，因此 CFG 的方向不一定对应任何分类器的梯度——这说明 CFG 是不可解释为分类器梯度攻击的纯生成方法。[^src-classifier-free-diffusion-guidance]

## 连续时间扩散框架

采用连续时间扩散训练（Song et al., 2021b; Kingma et al., 2021）：前向过程使用对数信噪比 $\lambda = \log(\alpha_\lambda^2 / \sigma_\lambda^2)$ 参数化，$\lambda$ 从 $[\lambda_\text{min}, \lambda_\text{max}]$ 中的双曲正割分布采样。使用 $\epsilon$-prediction 参数化和去噪得分匹配目标。方差超参数 $v$ 控制采样器的噪声注入量。[^src-classifier-free-diffusion-guidance]

## 实验结果

在 ImageNet 64×64 和 128×128 类别条件生成上验证：

- **64×64**：$w$ 从 0 到 4 扫参，FID 单调下降、IS 单调上升。最优 FID 在 $w=0.1\sim0.3$（FID=1.55），最优 IS 在 $w=4.0$（IS=250.4）。
- **128×128**：在 $w=0.3$ 时 FID=2.43，**超越**分类器引导的 ADM-G（FID=2.97）；在 $w=4.0$ 时 FID=21.53 + IS=421.03，同时超越 BigGAN-deep 的最佳 IS 水平。
- **$p_\text{uncond}$ 效应**：$\{0.1, 0.2, 0.5\}$ 中 $0.1$ 和 $0.2$ 表现相当，均显著优于 $0.5$。表明只需将少量模型容量分配给无条件生成任务即可获得有效引导。
- **采样步数**：$T=256$ 在质量和速度间最佳，但每个采样步需两次前向传播（条件+无条件），实际计算量相当于 $T=128$ 的分类器引导。[^src-classifier-free-diffusion-guidance]

## 局限性

1. **采样速度**：每次采样步需两次模型前向传播，使推理成本翻倍。作者建议后期注入条件可能缓解。
2. **多样性下降**：引导强度增大时样本多样性下降，在数据分布不平衡的应用中可能有问题。
3. **需要训练无条件模型**：在小类别空间可通过 $p(x) = \sum_c p(x|c)p(c)$ 求和避免，但在高维条件空间中不可行。[^src-classifier-free-diffusion-guidance]

## 影响

CFG 已成为现代扩散模型（Stable Diffusion、DALL-E 2、Imagen 等）的核心条件生成技术，几乎完全取代了分类器引导。该论文证明了纯生成扩散模型无需分类器即可实现引导，消除了对抗性攻击的可解释性疑虑。[^src-classifier-free-diffusion-guidance]

[^src-classifier-free-diffusion-guidance]: [[source-classifier-free-diffusion-guidance]]
