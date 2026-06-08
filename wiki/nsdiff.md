---
title: "NsDiff"
type: entity
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - ddpm
  - non-stationary
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 2
confidence: high
status: active
---

# NsDiff

**NsDiff** (Non-stationary Diffusion) 是首个将 [[location-scale-noise-model|Location-Scale Noise Model (LSNM)]] 整合到条件 DDPM 中进行概率时间序列预测的方法，由 Li et al. (IIT/中科院) 发表于 ICML 2025 Spotlight[^src-nsdiff]。

## 核心创新

NsDiff 从两个维度突破了传统 DDPM 在时序预测中的限制：

1. **端点分布扩展**：将 DDPM 的反向过程终点从 $\mathcal{N}(0, I)$ 推广到 $\mathcal{N}(f_\phi(X), g_\psi(X))$，同时建模数据的均值和方差[^src-nsdiff]。
2. **不确定性感知噪声调度**：前向噪声方差从固定的 $\beta_t I$ 扩展为 $\beta_t^2 g_\psi(X) + \beta_t \alpha_t \sigma_{Y_0}$，使得加噪过程能感知数据的不确定性水平[^src-nsdiff]。

这使得 NsDiff 在前序工作（[[timegrad|TimeGrad]]、CSDI、TMDM）构成统一的特例框架：TMDM 对应 $g_\psi(X)=I$ 的情形，TimeGrad 的 N(0,I) 去噪也是 LSNM 的退化形式[^src-nsdiff]。

## 架构

```
输入 X (历史序列 + 协变量)
    │
    ├──► f_φ(X)  — Non-stationary Transformer (均值估计器)
    │
    └──► g_ψ(X)  — 3层 MLP (hidden=512, ReLU, 方差估计器)
    │
    ▼
逆扩散过程: Y_T ∼ N(f_φ(X), g_ψ(X))
    │  T=20 步，线性 β 调度
    ▼
    Y_0 — 预测的多变量向量
```

方差估计器使用滑动窗口预训练提取真实 $\sigma_{Y_0}$（窗口=96），随后用 $\ell_2$ 损失训练 MLP 回归[^src-nsdiff]。推理时 $\sigma_{Y_0}$ 通过 Vieta 二次方程从预测的 $\sigma_\theta$ 反解[^src-nsdiff]。

## 性能

在 9 个真实数据集上以 CRPS 和 QICE 评估，对比 TimeGrad (ICML 2021)、CSDI (NeurIPS 2021)、TimeDiff (ICML 2023)、DiffusionTS (2024)、TMDM (ICLR 2024)[^src-nsdiff]。NsDiff 在除 Solar 外的所有数据集上达到 SOTA，其中高不确定性变化的数据集改进最大——Traffic 数据集（不确定性比=181.83）QICE 下降 66.3%[^src-nsdiff]。计算效率优于前序 SOTA TMDM（训练内存 68MB vs 221MB）[^src-nsdiff]。

## 与相关方法的关系

- **[[ddpm|DDPM]]**：NsDiff 继承 $L_{\text{simple}}$ 范式和 $\varepsilon$ 预测参数化，但将方差假设从固定单位方差推广为 LSNM[^src-nsdiff]
- **[[timegrad|TimeGrad]]**：首个时序扩散模型，但使用 $\mathcal{N}(0,I)$ 去噪起点，NsDiff 以 LSNM 取代了该限制[^src-nsdiff]
- **TMDM**：最近的 SOTA（ICLR 2024），使用 $\mathcal{N}(f_\phi(X), I)$ 去噪起点——是 NsDiff 在 $g_\psi(X)=I$ 时的特例[^src-nsdiff]
- **[[generative-time-series-forecasting|生成式时序预测]]**：NsDiff 是该范式下当前最强的扩散基线[^src-nsdiff]

## 局限性

1. 预训练+微调两阶段增加训练复杂性（虽端到端也可行但收敛稍慢）[^src-nsdiff]
2. 均值估计器默认使用相对重的 Non-stationary Transformer 增加了计算开销（但框架可替换）[^src-nsdiff]
3. Solar 等低不确定性变化数据集上优势不明显[^src-nsdiff]
4. 被 [[stats|StaTS]] (arXiv 2026) 在所有 CRPS 和 MAE 上超越，StaTS 以频谱轨迹调度+频率引导去噪在 8 个基准上取得 10.67%–17.43% CRPS 改进[^src-stats]

[^src-nsdiff]: [[source-nsdiff]]
[^src-stats]: [[source-stats]]
