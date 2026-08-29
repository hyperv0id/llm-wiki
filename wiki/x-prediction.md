---
title: "x-prediction"
type: concept
tags:
  - diffusion
  - prediction-target
  - manifold
  - training
created: 2026-05-13
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# x-prediction

x-prediction 是指扩散模型中让神经网络直接预测干净数据 $x$（而非噪声 $\epsilon$ 或速度 $v$）的参数化方式。虽然 x-prediction 最早可追溯到 DDPM 的原始代码[^src-back-to-basics-let-denoising-generative-models-denoise]，但长期以来未被作为主要预测目标使用。

## 三种预测目标

在扩散/流匹配模型中，给定 $z_t = t x + (1-t)\epsilon$ 和 $v = x - \epsilon$，存在三种可能的网络直接输出：

| 预测目标 | 网络输出 | 推导其余量 | 流形位置 |
|----------|----------|------------|----------|
| **x-prediction** | $x_\theta = \text{net}_\theta(z_t, t)$ | $\epsilon_\theta = (z_t - t x_\theta)/(1-t)$, $v_\theta = (x_\theta - z_t)/(1-t)$ | **on-manifold**（位于低维流形） |
| **ε-prediction** | $\epsilon_\theta = \text{net}_\theta(z_t, t)$ | $x_\theta = (z_t - (1-t)\epsilon_\theta)/t$, $v_\theta = (x_\theta - z_t)/(1-t)$ | off-manifold |
| **v-prediction** | $v_\theta = \text{net}_\theta(z_t, t)$ | $x_\theta = (1-t)v_\theta + z_t$, $\epsilon_\theta = z_t - t v_\theta$ | off-manifold |

三种预测目标 + 三种损失空间 = 九种合法组合（Tab. 1 in [^src-back-to-basics-let-denoising-generative-models-denoise]），所有组合都是有效的生成器，但在高维空间中性能差异巨大。

## 流形假设下的根本差异

根据流形假设，自然数据位于高维空间的低维流形上。x-prediction 与 ε-/v-prediction 的根本区别在于：

- **x-prediction 的输出是 on-manifold 的**：网络只需保留低维流形信息、过滤噪声。有限容量网络可以胜任，甚至在 bottleneck 设计下表现更好（因为 bottleneck 自然鼓励学习低维表示）。
- **ε-/v-prediction 的输出是 off-manifold 的**：噪声散布在整个高维空间，网络需要保留所有维度信息。当观测维度超过网络容量时（例如 768-d patch 但隐藏层仅 768-d），会灾难性失败。[^src-back-to-basics-let-denoising-generative-models-denoise]

## 历史

1. **DDPM 时期 (2020)**：原始 DDPM 代码中实现了 x-prediction，但发现 ε-prediction 效果更好，此后 ε-prediction 成为默认选择。[^src-back-to-basics-let-denoising-generative-models-denoise]
2. **v-prediction 引入 (2022)**：Salimans & Ho 在 Progressive Distillation 中引入 v-prediction，建立了三种预测目标与损失加权的理论联系。[^src-back-to-basics-let-denoising-generative-models-denoise]
3. **EDM 时代 (2022)**：Karras 等人提出的 EDM pre-conditioner 使用 $x_\theta = c_\text{skip} \cdot z_t + c_\text{out} \cdot \text{net}_\theta$，除非 $c_\text{skip} \equiv 0$，否则网络输出不是纯 x-prediction。实验表明 pre-conditioner 在高维下同样失败（但比纯 ε-/v-prediction 略好，因为 $t \to 0$ 时趋近 x-prediction）。[^src-back-to-basics-let-denoising-generative-models-denoise]
4. **JiT/Back-to-Basics (2025)**：Li & He 系统论证了 x-prediction 在高维空间的必要性，并提出 JiT——使用 x-prediction + 标准 ViT + 大 patch 的像素空间扩散模型。[^src-back-to-basics-let-denoising-generative-models-denoise]

## 实践建议

- **当 patch/输入维度 ≤ 网络隐藏维度时**：三种预测目标均可工作（如 CIFAR-10、低分辨率 ImageNet 64×64、大多数 latent 空间方法）
- **当 patch/输入维度 > 网络隐藏维度时**：**必须使用 x-prediction**，ε-/v-prediction 会灾难性失败
- **bottleneck 设计推荐**：x-prediction 下，将高维输入压缩到 32~512-d bottleneck 不仅无害，反而提升质量
- **推荐配置**：x-prediction + v-loss（Tab. 1(3)(a)），这是 JiT 采用的最优组合

## FM 指南的参数化体系（affine 路径）

[[source-flow-matching-guide|FM 指南]]（arXiv:2412.06264，2024-12）第 4.8 节在 affine 条件流 $\psi_t=\alpha_t x_1+\sigma_t x$（scheduler $(\alpha_t,\sigma_t)$）下给出与本页同构的参数化体系：边缘速度可由 velocity、$x_1$-prediction（去噪目标 $\mathbb{E}[X_1|X_t=x]$）、$x_0$-prediction（$\mathbb{E}[X_0|X_t=x]$）或 score 学习，四种参数化之间有闭式线性转换表（指南 Table 1）[^src-flow-matching-guide]。与本页扩散记号的对应：指南指出扩散文献的 noise-prediction/ε-prediction 就是 $x_0$-prediction，且高斯路径下 score $=-x_{0|t}/\sigma_t$、与 $x_0$-prediction 成比例[^src-flow-matching-guide]。

指南补充的两点口径：

- **奇点**：$x_1$-prediction 在 $t\to1$、$x_0$-prediction 在 $t\to0$ 处转换系数发散；理论上（条件期望存在时）为可去奇点、与系数奇点相消，但实践中连续的网络输出不能完美回归目标，转换时需用端点解析式 $u_0,u_1$ 处理[^src-flow-matching-guide]。这与本页流形论证互补：Li & He 论证的是输出空间性质（on-manifold），指南指出的是转换公式的数值性质。
- **scheduler 等价与训练后变换**：对固定耦合，所有 scheduler 在 $t=1$ 理论上给出相同采样结果；affine 路径支持训练后 scheduler 变换（scale-time 变换），可把已训练速度场适配到另一 scheduler（指南引 Karras et al. 2022、Shaul et al. 2023、Pokle et al. 2023）[^src-flow-matching-guide]。

## 与其他概念的关系

- [[diffusion-model]] — 扩散模型的整体框架，ε-prediction 是其标准参数化
- [[edm-design-space]] — EDM pre-conditioner 隐式偏离了纯 x-prediction
- [[jit|JiT]] — 首个系统依赖 x-prediction 的像素空间扩散模型
- [[elf|ELF]] — 同实验室的连续 DLM，同样使用 x-prediction 实现共享权重 denoiser-decoder
- [[edm-preconditioning]] — EDM 网络预处理技术，与 x-prediction 存在内在矛盾
- [[flow-matching]] — v-prediction 的数学框架，与 x-prediction 通过线性变换关联
- [[flow-matching-design-space]] — FM 指南的设计选择体系（参数化转换表所在框架）

[^src-back-to-basics-let-denoising-generative-models-denoise]: [[source-back-to-basics-let-denoising-generative-models-denoise]]
[^src-flow-matching-guide]: [[source-flow-matching-guide]]