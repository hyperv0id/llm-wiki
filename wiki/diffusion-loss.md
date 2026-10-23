---
title: "Diffusion Loss"
type: technique
tags:
  - diffusion
  - autoregressive-modeling
  - loss-function
  - image-generation
  - neurips-2024
created: 2026-10-23
last_updated: 2026-10-23
source_count: 2
confidence: medium
status: active
---

# Diffusion Loss

**Diffusion Loss** 是 MAR 论文提出的 per-token 概率分布建模方法：用一个小型去噪网络（MLP）加扩散过程表示连续值 token 的条件分布 p(x|z)，使自回归模型摆脱对 vector quantization（VQ）离散 tokenizer 的依赖[^src-2406-11838-mar]。

## 动机：自回归真正需要的是什么

离散 token 的自回归（如 GPT）之所以用 categorical cross-entropy，是因为离散空间天然对应 categorical 分布——损失函数与采样器（温度 + Gumbel-max/逆变换采样）都现成。论文的观察是：**必需的只是"一个损失函数 + 一个对应采样器"来建模 per-token 分布**，离散性只是让这件事变方便，并非概念前提[^src-2406-11838-mar]。由此，连续值 token 的自回归缺的不是 tokenizer，而是连续分布的损失/采样器——这就是 Diffusion Loss 补上的部分。

## 机制

设连续值目标 token 为 x ∈ R^d，自回归网络在该位置输出条件向量 z ∈ R^D[^src-2406-11838-mar]：

1. **损失**：$\mathcal{L}(z,x)=\mathbb{E}_{\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t|t,z)\|^2\right]$——标准 ε-prediction 去噪准则，但作用在每个 token 上而非整图/整序列；z 与噪声调度时间步嵌入相加后经 AdaLN 注入 MLP 的 LayerNorm 层。
2. **采样**：推理时从 $x_T\sim\mathcal{N}(0,I)$ 出发跑反向扩散得到 $x_0\sim p(x|z)$。
3. **训练细节**：去噪 MLP 为若干个 LN-linear-SiLU-linear 残差块（默认 3 块、宽 1024、21M 参数），与自回归模型联合训练，梯度经 z 回传；扩散过程沿用 iDDPM 的 cosine schedule（训练 1000 步、推理 100 步），每个图像对 t 采样 4 次以提高损失利用率；可选加入 variational lower bound 项。
4. **温度采样**：连续情形下温度 τ 通过缩放采样器的噪声项 $\sigma_t\delta$（或等价地将 $\epsilon_\theta$ 除以 τ）实现，作用等价于 categorical 分布的 $\mathrm{softmax}(Wz/\tau)$——作者报告 τ 对 FID/IS 的影响与离散自回归中的温度同等关键。

## 关键证据（作者报告，ImageNet 256×256、MAR-L 400 epochs）

- **去噪 MLP 可以很小**：宽 256→1536（2M→45M 参数）对应 w/ CFG FID 2.45→1.91，推理时间几乎不变（0.286→0.291 s/图，含 407M 的 Transformer）；默认 MLP 只占 MAR-L 约 5% 参数，采样约占 10% 运行时间[^src-2406-11838-mar]。
- **100 步采样足够**：FID/IS 在约 100 扩散步后趋于饱和[^src-2406-11838-mar]。
- **确定性回归不可行**：连续 token 直接用 L2 损失训练 MAR，FID > 100——没有分布建模就没有采样多样性[^src-2406-11838-mar]。
- **对 tokenizer 灵活**：可用于 VQ tokenizer 量化前的连续潜变量（7.82 vs cross-entropy 8.79）、stride 不匹配的 tokenizer（2×2 分组）、以及 Consistency Decoder 等其他架构的 tokenizer[^src-2406-11838-mar]。
- **与 GMM 路线对比**：GIVT 用预设混合数的高斯混合模型表达 token 分布，可表达的分布类型受限；扩散过程无此限制[^src-2406-11838-mar]。

## 定位与延伸

论文将 Diffusion Loss 定位为一种**参数化损失函数**（与对抗损失、感知损失同列），其"大模型建模 token 间依赖、小网络建模单 token 分布"的分工与 [[latent-diffusion-models|LDM]]/[[dit|DiT]] 式"扩散建模全部 token 联合分布"形成对照[^src-2406-11838-mar]。[[omnicast|OmniCast]] 在掩码生成框架中采用同构的 per-token 小 MLP 扩散头预测连续潜变量，用于天气预报[^src-omnicast]。

## 相关页面

- [[masked-autoregressive-model]] — 承载该损失的广义自回归框架
- [[source-2406-11838-mar]] — 论文摘要
- [[masked-generative-modeling]] — 掩码生成范式
- [[omnicast]] — per-token 扩散头在天气预测的应用
- [[ddpm]] — 扩散建模基础
- [[classifier-free-guidance]] — Diffusion Loss 原生支持 CFG（训练时 10% 替换 dummy 条件）

[^src-2406-11838-mar]: [[source-2406-11838-mar]]
[^src-omnicast]: [[source-omnicast]]
