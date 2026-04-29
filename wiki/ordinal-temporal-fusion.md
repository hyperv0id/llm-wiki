---
title: "Ordinal-Temporal Fusion"
type: technique
tags:
  - positional-encoding
  - rotary-position-embedding
  - temporal-modeling
  - fusion
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Ordinal-Temporal Fusion

**序数-时间融合** 是 SIREN-RoPE 的核心公式，将时间信号和序数位置信号通过可学习门控融合为统一的旋转角[^src-siren-rope]。

## 公式

$$\Theta_j(T_i, p_i) = \underbrace{f_\phi(T_i)_j \cdot \omega_j^s}_{\text{Temporal (SIREN)}} + \underbrace{p_i \cdot \theta_j \cdot \lambda}_{\text{Ordinal (scaled)}}$$

其中：
- $i \in \{0, \dots, C-1\}$：序列中的序数索引
- $p_i = i$：第 $i$ 个 item 的序数位置
- $f_\phi: \mathbb{R}^{d_t} \to \mathbb{R}^{d_k/2}$：双分支 SIREN 网络
- $\omega_j^s \in \mathbb{R}^{d_k/2}$：可学习每维频率缩放
- $\theta_j = base^{-2j/d_k}$：标准 RoPE 逆频率
- $\lambda \in \mathbb{R}$：可学习标量门控（初���化 1.0）[^src-siren-rope]

## 设计原则

### 1. 时间丰富性
- 捕获多尺度周期模式（日/周）���接从 $T_i$
- SIREN 分支自主发现未明确指定的隐藏周期

### 2. 序数保持
- 保留标准 RoPE 的单调近因衰减特性
- 保留平移等变性 $\Theta_j(T_i + \Delta, p_i + \Delta) = \Theta_{j-\Delta}(T_i, p_i)$

### 3. 自适应平衡
- $\lambda$ 通过端到端梯度下降学习
- 模型决定时间信号和序数信号各自的贡献权重[^src-siren-rope]

## 训练动态

实验观察：
- **无时间信号时**（标准 Ordinal RoPE）：$\lambda$ 始终保持在 1.0 附近
- **引入 SIREN-RoPE 后**：$\lambda$ 收敛到 0.044

这表明模型学会了主要依赖时间调制，序数位置几乎被完全替代[^src-siren-rope]

## 融合方式选择

| 融合方式 | 描述 | 效果 |
|----------|------|------|
| 加法融合 | $\Theta = \Theta_{temporal} + \Theta_{ordinal}$ | 简单有效 |
| 乘法融合 | $\Theta = \Theta_{temporal} \cdot \Theta_{ordinal}$ | 未采用 |
| 门控融合 | $\Theta = \Theta_{temporal} + \lambda \cdot \Theta_{ordinal}$ | SIREN-RoPE 采用 |

## 相关页面

- [[siren-rope]] — 主 entity 页面
- [[temporal-rotation]] — ���间旋转概念
- [[dual-branch-siren]] — 架构详情
- [[learnable-frequency-scaling]] — 可学习频率缩放

[^src-siren-rope]: [[source-siren-rope]]