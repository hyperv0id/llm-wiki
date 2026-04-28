---
title: "NTK-aware Interpolation"
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

# NTK-aware Interpolation

NTK-aware 插值是 RoPE 上下文扩展的一种方法，通过改变 RoPE 的 base 参数 $b$ 来分散插值压力，而非等比例拉伸所有维度[^src-yarn]。

## 动机

Position Interpolation (PI) 将所有 RoPE 维度等比例拉伸 $s$ 倍，导致高频分量被完全移除[^src-yarn]。根据 Neural Tangent Kernel (NTK) 理论，深度神经网络在低维输入上难以学习高频信息[^src-yarn]。RoPE 与 Fourier Features 高度相似，保留高频分量对位置编码至关重要[^src-yarn]。

## 数学定义

NTK-aware 插值通过修改 RoPE 的 base 参数实现[^src-yarn]：

$$b' = b \cdot s^{\frac{|D|}{|D|-2}}$$

其中：
- $b$：原始 base（通常为 10000）
- $s$：缩放因子 $L'/L$
- $|D|$：隐藏层维度

新的频率为 $\theta_d' = b'^{-2d/|D|}$。

## 工作原理

1. **高频少缩放**：高维度（大 $d$）的频率变化小，保留高频信息[^src-yarn]
2. **低频多缩放**：低维度（小 $d$）的频率变化大，适应更长上下文[^src-yarn]
3. **连续过渡**：中间维度平滑过渡，避免突变[^src-yarn]

## 优缺点

### 优势
- 无微调时性能远优于 PI[^src-yarn]
- 实现简单，仅需修改 base 参数[^src-yarn]
- 已被 Code Llama 等开源模型采用[^src-yarn]

### 劣势
- 最优 base 值需经验性搜索，增加调参成本[^src-yarn]
- 部分维度存在"越界"外推，微调效果不如 PI[^src-yarn]
- 理论缩放因子 $s$ 不能准确描述实际扩展倍数[^src-yarn]

## 与其他方法的关系

NTK-aware 是 [[ntk-by-parts-interpolation|NTK-by-parts]] 的前身，后者通过显式分段策略解决了 NTK-aware 的越界问题[^src-yarn]。

## 引用

[^src-yarn]: [[source-yarn]]