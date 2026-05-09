---
title: "DDPM"
type: entity
tags:
  - diffusion-models
  - generative-model
  - nips-2020
created: 2026-04-28
last_updated: 2026-05-09
source_count: 2
confidence: medium
status: active
---

# DDPM (Denoising Diffusion Probabilistic Models)

**DDPM** 是扩散模型领域里程碑式的工作，由 Jonathan Ho, Ajay Jain, Pieter Abbeel 于 2020 年发表在 NeurIPS。首次证明了扩散模型能够生成与 GAN 相媲美的高质量图像。

## 核心创新

1. **简化训练目标**：使用 $L_{\text{simple}}$ 预测噪声而非预测均值
2. **与得分匹配的等价性**：建立了扩散模型与去噪得分匹配之间的数学联系
3. **高质量样本**：CIFAR-10 达到 IS 9.46, FID 3.17

## 技术特点

- **T=1000 步扩散**：从纯噪声逐步去噪
- **U-Net 架构**：带自注意力和组归一化
- **噪声调度**：$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$

## 傅里叶空间视角

DDPM 的白噪声前向过程在傅里叶空间可写为 $y_t=\sqrt{\bar\alpha_t}y_0+\sqrt{1-\bar\alpha_t}n$，其中 $n$ 仍是复高斯白噪声。[^src-equal-snr] 若第 $i$ 个频率的数据方差为 $C_i$，则该频率的 SNR 为 $s_t^{\mathrm{DDPM}}(i)=\bar\alpha_t C_i/(1-\bar\alpha_t)$；自然图像等数据的 $C_i$ 随频率快速衰减，因此 DDPM 会更早、更快破坏高频分量。[^src-equal-snr]

这种频域 SNR 不均匀性诱导出 [[frequency-hierarchy-in-diffusion|低频先生成、高频后补全的层级]]，并可能使高频反向后验 $q(y_{t-1}\mid y_t)$ 更偏离单一高斯假设。[^src-equal-snr] [[equal-snr|EqualSNR]] 通过令 $\Sigma_{ii}=cC_i$ 消融这一层级，在标准 FID 上与 DDPM 大体持平，但高频谱统计更接近真实数据。[^src-equal-snr]

## 后续发展

- **[[score-based-sde|Score-Based SDE]]** (Song et al., ICLR 2021)：将 DDPM 重新解释为 VP SDE 的离散化
- **[[equal-snr|EqualSNR]]** (Falck et al., 2025)：从傅里叶 SNR 角度修改 DDPM 前向过程

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[ncsn|NCSN]] — DDPM 的重要前身
- [[score-based-sde|Score-Based SDE]] — 统一 SMLD 和 DDPM 的 SDE 框架
- [[score-based-generative-modeling]] — 基于分数的生成模型
- [[annealed-langevin-dynamics]] — 退火朗之万动力学
- [[frequency-hierarchy-in-diffusion]] — DDPM 的低频到高频隐式生成顺序

## 引用

[^src-ddpm]: [[source-ddpm]]
[^src-equal-snr]: [[source-equal-snr]]
