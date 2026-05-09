---
title: "EDM Design Space"
type: concept
tags:
  - diffusion
  - framework
  - sampling
  - training
created: 2026-04-28
last_updated: 2026-05-09
source_count: 2
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

## 链接

- [[edm]] — EDM 论文
- [[diffusion-model]] — 扩散模型基础
- [[ddpm]] — DDPM
- [[ncsn]] — NCSN
- [[score-based-sde]] — Score-Based SDE
- [[heun-sampler]] — Heun 采样器
- [[ edm-preconditioning]] — 预处理技术
- [[frequency-based-noise-control]] — 频域噪声控制（正交设计维度）

[^src-edm]: [[source-edm]]
[^src-2502-10236]: [[source-2502-10236]]