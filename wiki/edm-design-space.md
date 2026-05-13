---
title: "EDM Design Space"
type: concept
tags:
  - diffusion
  - framework
  - sampling
  - training
created: 2026-04-28
last_updated: 2026-05-13
source_count: 3
confidence: medium
status: active
---

# EDM Design Space

EDM 设计空间是 Karras 等人在 2022 年提出的统一框架[^src-edm]，将 VP、VE、iDDPM、DDIM 等扩散模型变体整合到一个公共框架下，使各组件正交化。

## 统一框架

EDM 将扩散模型采样过程表示为概率流 ODE[^src-edm]：

$$dx = \left[ \frac{\dot{s}(t)}{s(t)} x - \frac{s(t)^2 \dot{\sigma}(t)}{\sigma(t)} \nabla_x \log p\left(\frac{x}{s(t)}; \sigma(t)\right) \right] dt$$

其中 $\sigma(t)$ 为噪声调度，$s(t)$ 为信号缩放。

## 设计维度

| 维度 | 选项 |
|------|------|
| ODE 求解器 | Euler / Heun / RK45 |
| 时间步长调度 | 线性 / 指数 / EDM (ρ=7) |
| 噪声调度 σ(t) | √t / t / 常数 |
| 网络预处理 | cskip/cout/cin/cnoise |
| 噪声分布 | 均匀 / 对数正态 |
| 损失权重 | 1/σ² / EDM 权重 |

## 组件正交性

EDM 的核心观点是：上述各设计选择基本相互独立[^src-edm]，可以单独改进某一组件而不影响其他组件。这与此前将多个选择打包在一起的"紧耦合"设计形成对比。

## 与频域噪声控制的关系

[[frequency-based-noise-control|频域噪声控制]]与 EDM 设计空间是**正交**的：EDM 调整噪声量级 $\sigma(t)$ 和信号缩放 $s(t)$ 的数值，而频域噪声控制调整噪声在频域中的**形状**。两者可以组合使用——在 EDM 框架内对每个噪声水平 $\sigma$ 使用不同频域加权 $w_\sigma(\mathbf{f})$，实现噪声量级与频谱形状的独立调控。[^src-2502-10236]

## EDM 预处理器与 x-prediction 的矛盾

Li & He (2025) 指出，EDM 的 pre-conditioner 公式 $x_\theta(z_t, t) = c_\text{skip} \cdot z_t + c_\text{out} \cdot \text{net}_\theta(z_t, t)$ 意味着**除非 $c_\text{skip} \equiv 0$，否则网络的直接输出 $\text{net}_\theta$ 不是 x-prediction**。在高维空间中（如 JiT-B/16 的 768-d patch），pre-conditioner 同样会失败（FID 28~72 vs x-prediction 的 8.62），尽管比纯 ε-/v-prediction（FID 372+）略好——因为 $t \to 0$ 时 $c_\text{skip} \to 0$，$c_\text{out} \to 1$，趋近 x-prediction。[^src-2511-13720]

这一发现表明，pre-conditioner 设计假设网络需要在不同 $t$ 下输出不同量级的混合信号，但流形视角下，网络应该始终输出 on-manifold 的干净数据。详见 [[x-prediction]]。

## 链接

- [[edm]] — EDM 论文
- [[diffusion-model]] — 扩散模型基础
- [[ddpm]] — DDPM
- [[ncsn]] — NCSN
- [[score-based-sde]] — Score-Based SDE
- [[heun-sampler]] — Heun 采样器
- [[edm-preconditioning]] — 预处理技术
- [[frequency-based-noise-control]] — 频域噪声控制（正交设计维度）
- [[x-prediction]] — x-prediction，与 EDM pre-conditioner 存在内在矛盾
- [[jit|JiT]] — JiT，揭示 pre-conditioner 与 x-prediction 矛盾的工作

[^src-edm]: [[source-edm]]
[^src-2502-10236]: [[source-2502-10236]]
[^src-2511-13720]: [[source-back-to-basics-let-denoising-generative-models-denoise]]