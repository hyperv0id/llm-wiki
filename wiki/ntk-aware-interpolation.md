---
title: "NTK-aware Interpolation"
type: technique
tags:
  - rope
  - interpolation
  - context-extension
created: 2026-04-28
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# NTK-aware Interpolation

NTK-aware 插值是 RoPE 上下文扩展的一种方法，通过改变 RoPE 的 base 参数 $b$ 来分散插值压力，而非等比例拉伸所有维度[^src-yarn]。

## 动机

[[position-interpolation|Position Interpolation (PI)]] 把所有 $d/2$ 对频率统一除以 $s$、波长整体放大 $s$ 倍[^src-pi]；按波长分档的观点认为这等于把高频分量移除[^src-yarn]。PI 自身在 LLaMA 7B/13B 上把 2048 窗口扩到 32768（33B 到 16384、65B 到 8192），微调 1000 步，passkey 有效窗口 200 步后达标[^src-pi]。根据 Neural Tangent Kernel (NTK) 理论，深度神经网络在低维输入上难以学习高频信息[^src-yarn]。RoPE 与 Fourier Features 高度相似，保留高频分量对位置编码至关重要[^src-yarn]。

## 数学定义

NTK-aware 插值通过修改 RoPE 的 base 参数实现[^src-yarn]：

$$b' = b \cdot s^{\frac{|D|}{|D|-2}}$$

其中：
- $b$：原始 base（通常为 10000）
- $s$：缩放因子 $L'/L$
- $|D|$：隐藏层维度

新的频率为 $\theta_d' = b'^{-2d/|D|}$。

## 推导

要求两端：[^src-yarn]（Appendix A.2）

- 最高频维（$d=0$）压缩比为 1，频率不动；
- 最低频维（$d=|D|/2-1$）压缩比为 $1/s$，与 PI 的线性位置缩放等同。

设 $b'/b=s^{k}$，换 base 后第 $d$ 维的压缩比只由指数给出：

$$\frac{\theta_d'}{\theta_d}=\left(\frac{b'}{b}\right)^{-2d/|D|}=s^{-\frac{2kd}{|D|}}$$

$d=0$ 自动满足第一条；第二条要求 $\frac{2k(|D|/2-1)}{|D|}=1$，即 $k=\frac{|D|}{|D|-2}$，于是 $b'=b\cdot s^{|D|/(|D|-2)}$[^src-yarn]（Eq. 22）。原文 Eq. 21 把第二条写成 $b'^{(|D|-2)/|D|}=s\,b^{(|D|-2)/|D|}$，指数按「最后一个被旋转的维度 $d=|D|-2$」计；在复维下标下这一维是 $d=|D|/2-1$，$2(|D|/2-1)/|D|=(|D|-2)/|D|$，两种记法解出同一个 $k$。

两端锁定后，中间维的压缩比按 $d$ 连续变化：$d$ 小的一侧（短波长、高频）压缩少，$d$ 大的一侧（长波长、低频）压缩多[^src-yarn]。这就是它在 $d$ 上接近 NTK-by-parts 分段效果的原因，区别是这里由 base 的单一指数决定，而 NTK-by-parts 由 $r(d)$ 的阈值显式决定；分段规则的相位论证见 [[yarn#NTK-by-parts 为什么成立]]。

## 优缺点

### 优势
- 无微调时性能远优于 PI[^src-yarn]
- 实现简单，仅需修改 base 参数[^src-yarn]
- 已被 Code Llama 等开源模型采用[^src-yarn]

### 劣势
- 最优 base 值需经验性搜索，增加调参成本[^src-yarn]
- 高频维几乎不动，超过 $L$ 的位置在这些维上继续外推，因此它不是纯插值方案；微调效果不如 PI[^src-yarn]（Appendix A.2）
- 由于上面的外推，名义缩放因子 $s$ 不能准确反映实际扩展倍数，实践里 $s$ 要设得比目标倍数更大[^src-yarn]

## 与其他方法的关系

NTK-aware 是 [[ntk-by-parts-interpolation|NTK-by-parts]] 的前身，后者通过显式分段策略解决了 NTK-aware 的越界问题[^src-yarn]。

## 引用

[^src-pi]: [[source-position-interpolation-2306-15595]]
[^src-yarn]: [[source-yarn]]