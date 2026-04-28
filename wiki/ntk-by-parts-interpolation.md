---
title: "NTK-by-parts Interpolation"
type: technique
tags:
  - rope
  - interpolation
  - context-extension
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# NTK-by-parts Interpolation

NTK-by-parts 插值是 RoPE 上下文扩展的核心技术，根据各维度的波长 $\lambda_d$ 与上下文长度 $L$ 的关系，对不同维度采用不同的插值策略[^src-yarn]。

## 核心洞察

RoPE 并非纯粹的相对位置编码——当波长 $\lambda > L$ 时，该维度在预训练期间未完成完整旋转，实际上编码了绝对位置信息[^src-yarn]。因此：

- **$\lambda \ll L$（短波长）**：仅编码相对位置 → 不应插值[^src-yarn]
- **$\lambda \geq L$（长波长）**：编码绝对位置 → 必须插值[^src-yarn]
- **中间区域**：混合处理[^src-yarn]

## 数学定义

引入比率 $r(d) = \frac{L}{\lambda_d}$，表示第 $d$ 维在预训练上下文长度内的旋转圈数[^src-yarn]：

$$r(d) = \frac{L}{2\pi b^{2d/|D|}}$$

定义 ramp 函数 $\gamma(r)$[^src-yarn]：

$$\gamma(r) = \begin{cases} 0, & r < \alpha \\ 1, & r > \beta \\ \frac{r-\alpha}{\beta-\alpha}, & \text{otherwise} \end{cases}$$

NTK-by-parts 的插值函数为[^src-yarn]：

$$h(\theta_d) = \left(1 - \gamma(r(d))\right) \frac{\theta_d}{s} + \gamma(r(d)) \theta_d$$

## 参数选择

| 参数 | 含义 | LLaMA 推荐值 |
|------|------|-------------|
| $\alpha$ | 开始插值的阈值 | 1 |
| $\beta$ | 完全插值的阈值 | 32 |

- $r < 1$：波长 > 上下文长度 → 完全插值（$\gamma=0$，$h=\theta_d/s$）[^src-yarn]
- $r > 32$：波长 ≪ 上下文长度 → 不插值（$\gamma=1$，$h=\theta_d$）[^src-yarn]
- $1 \leq r \leq 32$：线性过渡[^src-yarn]

## 优势

- **精确控制**：显式区分相对位置维度和绝对位置维度[^src-yarn]
- **无越界外推**：所有维度严格在插值范围内[^src-yarn]
- **微调效果好**：优于 PI 和 NTK-aware[^src-yarn]
- **无微调也可用**：性能优于 PI[^src-yarn]

## 与 YaRN 的关系

NTK-by-parts 是 [[yarn|YaRN]] 的核心组成部分。YaRN 在此基础上增加了注意力温度缩放，进一步提升了长上下文性能[^src-yarn]。

## 引用

[^src-yarn]: [[source-yarn]]