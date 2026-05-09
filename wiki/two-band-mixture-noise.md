---
title: "两频段混合噪声"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - noise-schedule
created: 2026-05-09
last_updated: 2026-05-09
source_count: 2
confidence: high
status: active
---

# 两频段混合噪声（Two-Band Mixture Noise）

**两频段混合噪声**是[[frequency-diffusion|频域扩散]]中使用的具体噪声构造方式，将频谱分为低频段和高频段，分别施加带通滤波噪声后线性组合。[^src-2502-10236]

---

## 数学定义

### 频域坐标

令 $\mathbf{x} \in \mathbb{R}^{H \times W}$ 为空域图像。频域坐标 $\mathbf{f} = (f_x, f_y)$ 定义为：

$$f_x = \frac{k_x}{W},\quad f_y = \frac{k_y}{H},\quad k_x, k_y \in \mathbb{Z}$$

其中 $k_x, k_y$ 为整数索引，$H, W$ 为图像尺寸。[^src-2502-10236]

### 径向距离

频域中某点 $(f_x, f_y)$ 到中心的归一化径向距离：

$$d(f_x, f_y) = \sqrt{\left(f_x - \frac{1}{2}\right)^2 + \left(f_y - \frac{1}{2}\right)^2}$$

### 带通掩码

对给定频段 $[a, b]$ 的二进制掩码：

$$w(\mathbf{f}) = \mathbf{M}_{[a,b]}(f_x, f_y) = \begin{cases} 1, & a \leq d(f_x, f_y) \leq b \\ 0, & \text{otherwise} \end{cases}$$

### 频段受限噪声

频段 $[a, b]$ 内的噪声通过以下流程生成：

$$\boldsymbol{\epsilon}_{[a,b]} = \Re\left(\mathcal{F}^{-1}\left(\mathbf{N}_{\text{freq}} \odot \mathbf{M}_{[a,b]}\right)\right)$$

其中 $\mathbf{N}_{\text{freq}} = \mathbf{N}_{\text{real}} + i\mathbf{N}_{\text{imag}},\ \mathbf{N}_{\text{real}}, \mathbf{N}_{\text{imag}} \sim \mathcal{N}(0, \mathbf{I})$。[^src-2502-10236]

### 两频段线性组合

$$\boldsymbol{\epsilon}_f = \gamma_l\,\boldsymbol{\epsilon}_{[a_l,b_l]} + \gamma_h\,\boldsymbol{\epsilon}_{[a_h,b_h]}$$

约束条件：

$$\gamma_l + \gamma_h = 1,\quad \gamma_l, \gamma_h \geq 0$$

$$a_l < b_l \leq a_h < b_h,\quad [a_l,b_l] \cap [a_h,b_h] = \varnothing$$

实验中固定 $a_l = 0, b_h = 1, b_l = a_h = 0.5$，扫描 $\gamma_l \in [0.1, 0.2, \dots, 0.9]$ 和 $\gamma_h = 1 - \gamma_l$。[^src-2502-10236]

---

## 标准白噪声作为特例

当参数取以下值时，两频段混合退化为标准白噪声：

$$\gamma_l = \gamma_h = 0.5,\quad a_l = 0,\ b_l = 0.5,\ a_h = 0.5,\ b_h = 1$$

此时：

$$\boldsymbol{\epsilon}_f = 0.5 \cdot \boldsymbol{\epsilon}_{[0,0.5]} + 0.5 \cdot \boldsymbol{\epsilon}_{[0.5,1]} = \boldsymbol{\epsilon}$$

因为两个互补频段的带通噪声之和等于全频段噪声（即白噪声）。[^src-2502-10236]

---

## 参数控制

| 参数 | 范围 | 含义 | 默认值（标准扩散） |
|------|------|------|-------------------|
| $\gamma_l$ | $[0, 1]$ | 低频噪声权重 | 0.5 |
| $\gamma_h$ | $[0, 1]$ | 高频噪声权重（$\gamma_h = 1 - \gamma_l$） | 0.5 |
| $a_l$ | $[0, 0.5)$ | 低频段下界 | 0 |
| $b_l$ | $(0, 0.5]$ | 低频段上界 | 0.5 |
| $a_h$ | $[0.5, 1)$ | 高频段下界 | 0.5 |
| $b_h$ | $(0.5, 1]$ | 高频段上界 | 1 |

### 控制效果

- $\gamma_l > 0.5$：更多低频噪声 → 低频信息被更多破坏 → 模型被迫学习低频结构
- $\gamma_l < 0.5$：更多高频噪声 → 高频信息被更多破坏 → 模型被迫学习高频细节
- $\gamma_l = \gamma_h = 0.5$：标准扩散，各频率均匀

### 选择性学习的参数设置

当数据在频段 $[a_c, b_c]$ 被破坏时，训练参数设置为：

$$b_l = a_c,\quad a_h = b_c$$

这样：
- 低频段 $[a_l, b_l] = [0, a_c)$
- 高频段 $[a_h, b_h] = (b_c, 1]$
- 破坏频段 $[a_c, b_c]$ 被跳过[^src-2502-10236]

---

## 扩展方向

论文提出但未实验验证的扩展：[^src-2502-10236]

1. **更复杂的加权函数**：幂律加权 $w(\mathbf{f}) = \|\mathbf{f}\|^\alpha$ 和指数衰减加权 $w(\mathbf{f}) = \exp(-\beta\|\mathbf{f}\|^2)$ 作为带通混合的替代方案
2. **动态时变频域调度**：随扩散步数 $t$ 变化频域偏置，如先低频后高频（类似人眼视觉的渐进细节感知）

## 链接

- [[frequency-diffusion]] — 频域扩散
- [[frequency-based-noise-control]] — 频域噪声控制
- [[source-sagd]] — SAGD 完整版论文（bpm-SAGD 是其正式化）

## 引用

[^src-2502-10236]: [[source-2502-10236]]
[^src-sagd]: [[source-sagd]]