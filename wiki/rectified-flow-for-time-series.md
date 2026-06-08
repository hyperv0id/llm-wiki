---
title: "Rectified Flow for Time Series Generation"
type: concept
tags:
  - rectified-flow
  - flow-matching
  - time-series-generation
  - ode
  - efficient-sampling
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Rectified Flow for Time Series Generation

**Rectified Flow for Time Series** 是将 rectified flow 范式从图像/视频生成迁移到时间序列生成的应用。与扩散模型在时间序列中依赖数百步的弯曲 ODE/SDE 轨迹不同，rectified flow 学习从噪声到时间序列数据的直线输运路径，大幅降低采样成本[^src-flowts]。

## 与图像域 Rectified Flow 的差异

图像域的 rectified flow（如 InstaFlow）通常需要 reflow 迭代来拉直轨迹，而 FlowTS 直接在时间序列域中学习直线路径：

| 维度 | 图像 Rectified Flow | 时间序列 Rectified Flow |
|------|---------------------|------------------------|
| 数据维度 | 高维像素空间 | 中等维度时序空间 ($\mathbb{R}^{\ell \times d}$) |
| 架构 | U-Net / DiT | 编码器-解码器 Transformer |
| 采样步数 | 1-4 步 (reflow 后) | 10-100 步，30 步即可 SOTA |
| 条件化 | 文本/类别引导 | 观测值替换 + 迭代精炼 |
| 先验结构 | 各向同性高斯 | 支持趋势-季节分解注入 |

## 核心机制

### 直线 ODE 输运

给定时间序列 $Z_1 \sim \pi_1$（目标分布）和 $Z_0 \sim \mathcal{N}(0, I)$（噪声），rectified flow 学习 ODE[^src-flowts]：

$$\frac{dZ_t}{dt} = v(Z_t, t), \quad Z_t = t Z_1 + (1-t) Z_0$$

通过最小二乘回归直接预测方向向量 $(Z_1 - Z_0)$：

$$\mathcal{L} = \mathbb{E}_{t \sim \text{Logit-Normal}}\left[\|(Z_1 - Z_0) - G(Z_t, t)\|^2\right]$$

### 无条件到条件的无缝迁移

这是时间序列 rectified flow 的关键优势：无条件训练模型在推理时可直接用于条件生成（预测/插补），无需重新训练[^src-flowts]。条件生成时用观测值替换对应位置：

$$\hat{Z}_1 \leftarrow \hat{Z}_1 \odot (1 - M) + Z_1 \odot M$$

然后通过 ODE 积分向前推进。这在图像域中通常需要 classifier-free guidance 等额外机制。

### 时间序列特有的归纳偏置

与通用 rectified flow 不同，时间序列版本集成了[^src-flowts]：
- **趋势-季节分解**：Trend Synthetic Layers + Fourier Synthetic Layers 显式建模周期和长期模式
- **Attention Registers**：可学习 token 作为全局上下文聚合器
- **RoPE**：旋转位置编码捕获时序相对位置

## 效率优势

- **训练**：FlowTS 以 2,500 迭代超越 Diffusion-TS 的 10,000 迭代[^src-flowts]
- **推理**：30 步采样 vs 扩散模型的 100-200 步[^src-flowts]
- **理论**：直线路径是传输成本最小化的测地线，避免弯曲轨迹的数值误差累积[^src-flowts]

## 与相关方法对比

| 方法 | 生成框架 | TS-Specific 归纳偏置 | 无条件→条件迁移 |
|------|---------|---------------------|-----------------|
| **FlowTS** | Rectified Flow (ODE) | Trend-Season + RoPE + Register | ✓ (无需重训) |
| Diffusion-TS | DDPM | Trend-Season | ✗ (需条件训练) |
| TSFlow | CFM + GP Prior | GP 先验核函数 | 部分 (CPS + guidance) |
| DiTS | Flow Matching | AdaLN + 双流注意力 | ✗ (独立架构) |
| CSDI | Score-based Diffusion | 双向自注意力 | ✗ (条件训练) |

## 相关页面

- [[rectified-flow]] — Rectified Flow 理论基础
- [[flowts]] — FlowTS 模型实体
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[diffusion-models]] — Diffusion Models

[^src-flowts]: [[source-flowts]]
