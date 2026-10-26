---
title: "正弦位置编码"
type: technique
tags:
  - position-encoding
  - sinusoidal
  - transformer
  - absolute-position
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# 正弦位置编码

正弦位置编码（sinusoidal positional encoding）是 Transformer 采用的固定位置编码，通过一组不同频率的正弦与余弦函数生成位置向量，并与 token 嵌入逐元素相加，使不含循环与卷积结构的模型能够利用序列顺序（§3.5）[^src-attention-1706-03762]。

## 公式与频率结构

$$\begin{aligned}
PE(pos,\,2i) &= \sin\!\left(\omega_i\,pos\right),\quad PE(pos,\,2i+1) = \cos\!\left(\omega_i\,pos\right) && \text{// } \omega_i = 10000^{-2i/d_{\text{model}}}\ \text{——每两维共用一条频率，sin/cos 相位正交}
\end{aligned}$$

在维度符号上，以 $pos\in\{1..N\}$ 表示序列位置，$i\in\{0..d_{\text{model}}/2-1\}$ 表示频率序号，$d_{\text{model}}$ 表示嵌入宽度。位置向量形状为 $[N, d_{\text{model}}]$，与嵌入同宽，逐元素相加后直接送入编码器与解码器栈底，层内不再重复注入（§3.5）[^src-attention-1706-03762]。位置编码的每一维都是一条正弦曲线，$i$ 越大则频率越低。

波长 $\lambda_i=2\pi/\omega_i=2\pi\cdot10000^{2i/d_{\text{model}}}$ 构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数（§3.5）[^src-attention-1706-03762]：$i=0$ 处波长约 6.28，相邻位置的相位差约 1 弧度，对细粒度位移敏感；$i=d_{\text{model}}/2-1$ 处波长约 $6.2\times10^{4}$（$10000\cdot2\pi$ 为其上界），跨越整个训练长度几乎保持不变，起区分粗粒度区间的作用。同一对维度满足 $\sin^2+\cos^2=1$，编码值域固定在 $[-1,1]$，与嵌入相加不会改变嵌入的数值尺度。

## 关键推导：固定位移对应固定的二维旋转

对第 $i$ 对维度，把 $pos+k$ 展开：

$$\begin{aligned}
PE(pos+k,\,2i) &= \sin(\omega_i pos)\cos(\omega_i k)+\cos(\omega_i pos)\sin(\omega_i k) && \text{// 和角公式，将 } pos \text{ 与位移 } k \text{ 分离}\\
PE(pos+k,\,2i+1) &= \cos(\omega_i pos)\cos(\omega_i k)-\sin(\omega_i pos)\sin(\omega_i k) && \text{// 同一对的 cos 分量，同样仅依赖 } \omega_i k \text{ 与 } \omega_i pos
\end{aligned}$$

将位置向量按行向量书写（$PE_{pos}\in\mathbb{R}^{1\times d_{\text{model}}}$）并表示为矩阵形式后，位移 $k$ 仅出现在与 $pos$ 无关的 $2\times2$ 矩阵中：

$$\begin{aligned}
\begin{bmatrix}PE(pos+k,2i) & PE(pos+k,2i+1)\end{bmatrix} &= \begin{bmatrix}PE(pos,2i) & PE(pos,2i+1)\end{bmatrix}\underbrace{\begin{bmatrix}\cos(\omega_i k) & -\sin(\omega_i k)\\ \sin(\omega_i k) & \cos(\omega_i k)\end{bmatrix}}_{R(\omega_i k)} && \text{// } [1, \bcancel{2}]\times[\bcancel{2}, 2]\to[1, 2]\ \text{——行向量右乘旋转矩阵，转角只由位移 } k \text{ 决定}\\
PE_{pos+k} &= PE_{pos}M_k,\quad M_k=\operatorname{diag}\!\left(R(\omega_0 k),\dots,R(\omega_{d_{\text{model}}/2-1}k)\right) && \text{// } [1, \bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}}, d_{\text{model}}]\to[1, d_{\text{model}}]\ \text{——整条编码乘一次固定的分块旋转}
\end{aligned}$$

论文将这一性质作为选择正弦形式的依据：对任意固定位移 $k$，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数，模型因此容易学到按相对位置关注（§3.5）[^src-attention-1706-03762]。上述分块旋转由和角公式直接写出，$M_k$ 不含可学习参数，也不随序列长度变化。

相对位移在编码空间中对应同一套旋转，不同 $k$ 仅改变各频率分量的转角 $\omega_i k$：低频分量转角小、高频分量转角大，一个线性变换即可将「距离 $k$」编码为跨频段的相位差[^src-attention-1706-03762]。该性质是论文提出「易学相对位置」假设的依据，论文未进一步检验训练后的注意力是否实际利用了这些旋转。

## 证据

Table 3 行 (E) 将正弦编码替换为学习式位置嵌入后，EN-DE newstest2013 dev BLEU 得分 25.7，与正弦 base 配置的 25.8 表现接近（§3.5、§6.2）[^src-attention-1706-03762]。选择正弦版本的理由是其可能外推至训练未见过的更长序列（§3.5）[^src-attention-1706-03762]。

## 边界

该编码本身表达绝对位置，其相对位置性质体现为线性可表示性，而非直接给出相对位置向量。关于外推能力，论文仅给出了动机陈述，未开展超出训练长度的实验，且同一节中与学习式嵌入的对照也仅在训练长度内评测（§3.5、Table 3）[^src-attention-1706-03762]。此外，位置信息只在两个栈底注入一次，层内没有位置信号，后续架构则将位置处理移入注意力打分（见 [[alibi]]、[[rope]]）。

## 相关页面

相关内容中，[[transformer-architecture]] 展示了位置向量在整体数据流中的位置，[[variant-frequency-positional-encoding]] 涉及将波长换成物理周期以用于多变量时空输入，[[learnable-patch-position-encoding]] 介绍学习式 patch 位置嵌入，[[adaptive-positional-encoding]]、[[generalized-positional-encoding-framework]] 涵盖相对位置编码的后续形式与统一框架，[[source-attention-is-all-you-need-1706-03762]] 则为论文摘要页。

---

[^src-attention-1706-03762]: [[source-attention-is-all-you-need-1706-03762]]
