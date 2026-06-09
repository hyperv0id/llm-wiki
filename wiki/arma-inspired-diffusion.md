---
title: "ARMA-Inspired Diffusion"
type: concept
tags:
  - diffusion-models
  - time-series
  - forecasting
  - arma
  - generalized-diffusion
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# ARMA-Inspired Diffusion

**ARMA-Inspired Diffusion** 指把经典 Auto-Regressive Moving Average (ARMA) 理论作为扩散式时间序列预测的设计原理：将时间序列视为前序数据点的连续序列演化加随机噪声的链式过程，并据此重塑扩散过程，使其建模序列演化而非单纯去噪[^src-armd]。该理念由 [[armd|ARMD]]（AAAI 2025）首次系统化为"连续序列扩散" (continuous sequential diffusion) 预测范式[^src-armd]。

## ARMA 回顾

ARMA 把当前值表示为自回归 (AR) 与移动平均 (MA) 两部分之和[^src-armd]：

$$x_t=\underbrace{\phi_1 x_{t-1}+\cdots+\phi_p x_{t-p}}_{\text{AR：过去数据点的线性组合}}+\underbrace{\theta_1\epsilon_{t-1}+\cdots+\theta_q\epsilon_{t-q}+\epsilon_t}_{\text{MA：过去误差(噪声)的线性组合}}.$$

AR 分量擅长捕捉长期趋势，MA 分量擅长处理突变与显著噪声[^src-armd]。

## 与扩散过程的映射

ARMD 用 ARMA 为其[[sliding-window-diffusion|滑动窗口扩散]]与[[distance-based-devolution|线性去演化]]提供理论依据[^src-armd]：

- **MA ↔ 前向演化**：滑动步 $X^t_{1-t:T-t}\to X^{t+k}_{1-t-k:T-t-k}$ 可视为按 ARMA 假设向序列注入噪声，移位序列中每个点 $x_i$ 携带来自 $i{+}1:i{+}k$ 时间步引入的噪声[^src-armd]。
- **AR ↔ 反向去演化**：线性去演化网络把每个点建模为前 $k$ 个时间步的线性组合，对应 AR 分量[^src-armd]。

作者主张这种与 ARMA 的一致性为扩散预测器的有效性提供了理论支撑[^src-armd]。

## 意义

ARMA-inspired diffusion 把"扩散即生成噪声→数据"的视角，替换为"扩散即模拟时间序列演化"，使扩散采样过程与预测目标天然对齐，并把统计学经典 (ARMA) 与现代生成模型 (扩散) 接合[^src-armd]。

[^src-armd]: [[source-armd]]

