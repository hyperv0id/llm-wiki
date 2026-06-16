---
title: "Source: Foundations of Schrödinger Bridges for Generative Modeling"
type: source-summary
tags:
  - schrödinger-bridge
  - generative-modeling
  - optimal-transport
  - diffusion-models
  - tutorial
created: 2026-06-16
last_updated: 2026-06-16
source_count: 0
confidence: high
status: active
---

# Source: Foundations of Schrödinger Bridges for Generative Modeling

**作者**: Sophia Tang (University of Pennsylvania)  
**arXiv**: [2603.18992](https://arxiv.org/abs/2603.18992)  
**日期**: 2026-03 | **篇幅**: 220 页

## 概述

本文是一份自包含的综合性教程，系统发展 Schrödinger bridge (SB) 的数学基础，作为**现代生成建模的统一框架**——涵盖 diffusion models、score-based models 和 flow matching。从 optimal transport 和 stochastic calculus 第一性原理出发，建立 static 和 dynamic SB 公式，连接 stochastic optimal control，给出多种 bridge 构造方法、各类变体及实用的生成建模算法。

## 结构

| 章节 | 内容 |
|------|------|
| §1 Static SB | Monge–Kantorovich OMT → EOT → static SB + Sinkhorn algorithm |
| §2 Dynamic SB | Path measures, Fokker-Planck, Feynman-Kac, Girsanov, Hopf-Cole transform |
| §3 SOC Formulation | HJB equation, value function, RE/CE/variance losses, SB-SOC |
| §4 Building Bridges | 六种构造法 (conditional bridges, time reversal, FBSDE, h-transform, IMF, interpolants) |
| §5 Variations | Gaussian, generalized, multi-marginal, unbalanced, branched, fractional SB |
| §6 Generative Modeling | Likelihood training, DSBM, [SF]²M, adjoint matching |
| §7 Discrete State Space | CTMC, discrete SB, discrete SOC, DDSBM |
| §8 Applications | Image translation, single-cell dynamics, Boltzmann sampling |

## 核心公式

- **Static SB**: $\pi_{0,T}^\star = \arg\min_{\pi\in\Pi} \mathrm{KL}(\pi\|q)$，解 $\pi^\star = e^{\varphi+\hat{\varphi}-c}\pi_0\otimes\pi_T$
- **Dynamic SB**: $\mathbb{P}^\star = \arg\min \mathrm{KL}(\mathbb{P}\|\mathbb{Q})$ s.t. $p_0=\pi_0, p_T=\pi_T$
- **Hopf-Cole**: $\psi_t=\log\varphi_t$, $p_t^\star=\varphi_t\hat{\varphi}_t$, $u^\star=\sigma_t\nabla\log\varphi_t$
- **HJB-FP**: $\partial_t\psi + \frac{\sigma^2}{2}\|\nabla\psi\|^2 + \langle\nabla\psi,f\rangle = -\frac{\sigma^2}{2}\Delta\psi$
- **Path RND**: $\frac{d\mathbb{P}^{\tilde{u}}}{d\mathbb{P}^u} = \exp(-\frac{1}{2}\int_0^T\|(\tilde{u}-u)\|^2 dt + \int_0^T(\tilde{u}-u)^\top dB_t^u)$

## 与 Wiki 的关系

本文作为基础参考，将 wiki 中已有的 [[diffusion-models]]、[[flow-matching]]、[[optimal-transport]]、[[score-based-generative-modeling]] 等概念统一在 SB 框架下，并引入大量新概念。

[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
