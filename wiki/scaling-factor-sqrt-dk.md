---
title: "缩放因子 1/√dₖ"
type: concept
tags:
  - transformer
  - attention
  - numerical-stability
  - softmax
  - variance-control
created: 2026-05-04
last_updated: 2026-10-26
source_count: 4
confidence: medium
status: active
---

# 缩放因子 $1/\sqrt{d_k}$

缩放因子 $1/\sqrt{d_k}$ 是 Scaled Dot-Product Attention 中的核心数值稳定性设计，在点积结果进入 Softmax 之前将其方差从 $d_k$ 归一化至 1，防止高维条件下 Softmax 饱和与梯度消失[^src-bluuuuue-scaling-factor-intuition]。

## 数学原理

### 点积方差膨胀

设 Query 向量 $Q$ 与 Key 向量 $K$ 各分量独立且服从 $N(0,1)$。点积 $Z = \sum_{i=1}^{d_k} q_i k_i$ 涉及 $d_k$ 个乘积项求和。每个乘积项 $q_i k_i$ 的方差为 $Var(q_i) \cdot Var(k_i) = 1$（利用独立零均值变量的乘积方差性质）。因此[^src-bluuuuue-scaling-factor-intuition]：

$$Var(Z) = d_k, \quad \sigma_Z = \sqrt{d_k}$$

$d_k$ 越大，点积值的分布越分散。当 $d_k = 64$ 时，点积的典型取值范围已在正负数十的量级[^src-bluuuuue-scaling-factor-intuition]。

### Softmax 饱与梯度消失

Softmax 函数 $\sigma(\mathbf{z})_i = e^{z_i} / \sum e^{z_j}$ 的偏导数为[^src-bluuuuue-scaling-factor-intuition]：

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

当输入方差极大时，最大分量经指数运算占据绝对主导，输出趋近 One-hot。此时 $\sigma_i(1 - \sigma_i) \to 0$，梯度消失——参数 $W_Q$ 和 $W_K$ 无法通过误差信号校正[^src-bluuuuue-scaling-factor-intuition]。

饱和导致的双重后果：
1. **信息表征层面**：注意力退化为单点聚焦，丧失多位置关注能力
2. **梯度传导层面**：Softmax 偏导数趋近于零，注意力头不再接收有效学习信号[^src-bluuuuue-scaling-factor-intuition]

### 缩放的数学推导

将点积除以 $\sqrt{d_k}$ 后[^src-bluuuuue-scaling-factor-intuition]：

$$Var\left(\frac{Z}{\sqrt{d_k}}\right) = \frac{1}{d_k} Var(Z) = \frac{1}{d_k} \cdot d_k = 1$$

方差归一化后，点积值约 95% 落在 $[-2, 2]$ 范围内，Softmax 具有足够的非线性区分度且避开饱和区[^src-bluuuuue-scaling-factor-intuition]。

## 原始出处（§3.2.1）

缩放因子随 Scaled Dot-Product Attention 一起提出，论证写在脚注 4：假设 $q$、$k$ 的各分量是独立随机变量、均值 0、方差 1，则点积 $q\cdot k$ 的均值为 0、方差为 $d_k$[^src-attention-1706-03762]。同节的对照是加法注意力——$d_k$ 小时两种注意力表现接近，$d_k$ 大时未缩放的加法注意力反超，论文据此把差异归到点积幅度而非交互函数形式；点积注意力实现更快、更省空间，因为可直接用矩阵乘法[^src-attention-1706-03762]。后续实验的默认尺度是 $h=8$、$d_k=d_v=64$（§3.2.2），完整数据流见 [[transformer-architecture]][^src-attention-1706-03762]。

## 三个数学特性

### 1. 恰好 $1/\sqrt{d_k}$ 而非 $1/d_k$

若除以 $d_k$，方差变为 $d_k \times (1/d_k)^2 = 1/d_k$，标准差 $\sqrt{1/d_k}$。高维条件下方差趋于零——点积被过度压缩，Softmax 输出趋于均匀分布，注意力失去聚焦能力。$1/\sqrt{d_k}$ 恰好将方差稳定在 1，避免了**欠缩放**与**过缩放**两种失效模式[^src-bluuuuue-scaling-factor-intuition]。

### 2. 保持相对排序

缩放操作 $f(x) = x/c$ 是单调递增函数，argmax 运算结果具有缩放不变性。唯一改变的是概率分布在 One-hot 与均匀分布之间的位置——属于纯数值稳定性修正，而非表征层面的改变[^src-bluuuuue-scaling-factor-intuition]。

### 3. 初始化阶段的不可替代性

在训练初期模型尚未学会调整 $Q$ 和 $K$ 的范数时，缩放因子防止因纯粹的数值尺度问题陷入不可逆的饱和状态。层归一化与残差连接使每层注意力输入的统计特性在训练中保持相对稳定，因此缩放因子对深层结构中的注意力稳定性具有持续贡献[^src-bluuuuue-scaling-factor-intuition]。

## 与位置项的共用缩放

[[relative-position-representation|相对位置向量]]加到 key 之后，$1/\sqrt{d_z}$ 仍统一乘在「内容项 + 位置项」之和外面[^src-shaw-2018-relative-position]：

$$e_{ij}=\frac{1}{\sqrt{d_z}}\Big(x_iW^Q\big(x_jW^K\big)^{\top}+x_iW^Q\big(a^K_{ij}\big)^{\top}\Big)$$

边向量与被加宽的 key 内积共用同一个缩放因子；只缩放内容项会让位置项的相对量级随 $d_z$ 与边向量范数漂移，而边向量的量级由训练中的 $w^K$ 自由决定，没有独立的斜率参数可调[INFERENCE]。

## 学习式尺度：把常数换成可学习标量

QK-Norm 沿 head 维对 query 与 key 做 ℓ2 归一化后乘一个可学习标量 $g$，用乘法代替这里的除法，$g$ 初始化为 $\log_2(L^2-L)$，$L$ 为训练集源/目标序列长度的 97.5 分位[^src-henry-2020-qk-norm]。两者钉的不是同一个量：$1/\sqrt{d_k}$ 按「分量独立、零均值、单位方差时点积方差为 $d$」把方差压到 1[^src-bluuuuue-scaling-factor-intuition]；归一化之后打分退化为余弦相似度，被 Cauchy–Schwarz 限制在 $[-1,1]$、与 $d$ 无关，尺度改由 $g$ 决定可取范围。这一族的中间形态是 ScaleNorm——沿 embedding 维归一 Q、K、V 后乘一个可学习标量，标量初值恰好取 $1/\sqrt{d}$，与本页这个常数同值[^src-henry-2020-qk-norm]。推导与实验见 [[qk-norm]]。

## 在后续架构中的保留

该缩放因子在 VLA（视觉-语言-动作模型）和世界模型等后续架构的注意力模块中被完整保留——无论是 Cross-Attention 还是 Causal Attention[^src-bluuuuue-scaling-factor-intuition]。其不可替代性来自所解决问题的结构性质：只要注意力沿用"高维向量内积施加 Softmax"这一基本形式，点积方差膨胀就是必然发生的数学事实。

[[multi-query-attention]] 的公式把 $1/\sqrt{d_k}$ 折算进 $P_q$、$P_k$ 后不再显式写出，维度与数值都不变，只是文面省略（§2.2）[^src-shazeer-2019-mqa]。

## 与相关概念的关系

| 概念 | 与缩放因子的关系 |
|------|-----------------|
| [[attention-entropy-collapse]] | 缩放因子防止方差膨胀导致的熵崩溃，是根本性的预防机制 |
| [[attention-logit-explosion]] | 缩放因子控制初始化阶段的 logit 尺度，防止 Q/K 范数增长的数值后果 |
| [[attention-temperature-scaling]] | 温度缩放是 $1/\sqrt{d_k}$ 的推广——在 RoPE 上下文扩展中调节 $t$ 值 |
| [[key-normalization]] | 键归一化从另一角度解决注意力不稳定——约束 K 范数而非缩放点积 |
| [[qk-norm]] | 把 $1/\sqrt{d_k}$ 换成乘在余弦打分上的可学习标量，初值由训练序列长度决定 |
| [[entropy-boundedness]] | 缩放因子确保注意力熵远离零（防止饱和）和远离最大值（防止均匀化） |

## 引用

[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]
[^src-attention-1706-03762]: [[source-attention-is-all-you-need-1706-03762]]
[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]
<<<<<<< HEAD
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
=======
[^src-henry-2020-qk-norm]: [[source-henry-2020-qk-norm]]
>>>>>>> cd8b2f8 (ingest Query-Key Normalization for Transformers)
