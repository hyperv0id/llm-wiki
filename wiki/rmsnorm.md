---
title: "RMSNorm（Root Mean Square Layer Normalization）"
type: technique
tags:
  - normalization
  - transformer
  - training-efficiency
  - layer-normalization
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# RMSNorm（Root Mean Square Layer Normalization）

LayerNorm 每步计算均值与方差两个统计量，均值用于 re-centering。RMSNorm 保留 re-scaling 不变性、去掉 re-centering，只按均方根归一化[^src-zhang-2019-rmsnorm]。去除该项的前提是它并未带来收益：RNNSearch 训练 loss 到 7.0 时，LayerNorm 在同训练步数下已到 5.4，同训练时间下只到 5.9，多出的每步开销抵消了更快收敛换来的净效率（Figure 1）[^src-zhang-2019-rmsnorm]。

## 定义与数据流

通用写法为 $\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot g$，其中 $x$ 是被归一化向量[^src-zhang-2019-rmsnorm]；后文沿用论文记号，将其记作 summed inputs $a$。逐元素写法见 Stage 2。

**维度符号**：$m$ 为单样本输入维；$n$ 为该层神经元数，即被归一化的 summed inputs 维，也是增益 $g$ 的长度；$B$ 为单次前向处理的样本位置数（batch $\times$ 序列位置）；$k$ 为 pRMSNorm 估计 RMS 所用分量数。论文按单样本 $x\in\mathbb{R}^{m}$ 书写，§4.1 说明等式可扩展至 batch 输入，展开至 $[B,n]$ 的写法标注 [INFERENCE]。

### Stage 1 · 求统计量（LayerNorm 两个、RMSNorm 一个）

LayerNorm 先求均值再求方差，方差依赖中心化后的 $a$，两步无法合成一次规约（式 2–3）：

$$\begin{aligned}
\mu &= \frac{1}{n}\sum_{i=1}^{n}a_i && \text{// } [B,n]\xrightarrow{\text{沿 } n \text{ 维求均值}}[B,1]\ \text{——第一个统计量}\\
\sigma &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(a_i-\mu\right)^{2}} && \text{// } [B,n]\text{ 与 }[B,1]\xrightarrow{\text{沿 } n \text{ 维求方差}}[B,1]\ \text{——第二个统计量，依赖前一步的 }\mu
\end{aligned}$$

RMSNorm 只留一个与均值无关的量（式 4）：

$$\begin{aligned}
\mathrm{RMS}(a) &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^{2}} && \text{// } [B,n]\xrightarrow{\text{沿 } n \text{ 维求均方}}[B,1]\ \text{——一个标量顶替 LayerNorm 的 } \mu \text{ 与 } \sigma
\end{aligned}$$

两者满足 $\mathrm{RMS}(a)^{2}=\sigma^{2}+\mu^{2}$，故 $\mathrm{RMS}(a)\geq\sigma$：均值非零时 RMSNorm 除数更大，输出与 LayerNorm 不同；均值恰为零时两者逐元素相等（§4）[^src-zhang-2019-rmsnorm]。实现中开方前加数值稳定项 $\epsilon$，式 4 省略该项。

### Stage 2 · 缩放并接回激活

$$\begin{aligned}
\bar{a}_i &= \frac{a_i}{\mathrm{RMS}(a)}\,g_i && \text{// } [B,n]/[B,1]\odot[n]\to[B,n]\ \text{——除数按样本广播，} g \text{ 逐神经元重新标定}\\
y_i &= f\!\left(\bar{a}_i+b_i\right) && \text{// } [B,n]+[n]\to[B,n]\ \text{——偏置 } b \text{ 与激活沿用式 1，归一化自身不引入额外仿射偏置}
\end{aligned}$$

相比 LayerNorm 有三处差异：不减均值；分母为均方根而非标准差；归一化后仅保留增益 $g$（初始化为 1），式 2 中的 $b$ 即式 1 的神经元偏置且位置未变，归一化只作用于 $a$，不新增偏置项（式 2、式 5）。

### Stage 3 · 只在前 $p\%$ 分量上估计 RMS（pRMSNorm）

$$\begin{aligned}
k &= \lceil n\cdot p\rceil,\quad \mathrm{RMS}(a)\approx\sqrt{\frac{1}{k}\sum_{i=1}^{k}a_i^{2}} && \text{// } [B,n]\xrightarrow{\text{取前 } k \text{ 个分量}}[B,k]\xrightarrow{\text{沿 } k \text{ 维求均方}}[B,1]\ \text{——规约维从 } n \text{ 缩到 } k
\end{aligned}$$

子集上的 $\mathrm{RMS}$ 仍对缩放因子一次齐次，式 6 的线性原样保留，不变性质与 RMSNorm 一致（Table 1）[^src-zhang-2019-rmsnorm]。$\mathrm{RMS}(a)$ 因此是有偏估计，$k$ 偏小时梯度易爆（§5 记作 small $m$），$p=6.25\%$ 能稳定收敛[^src-zhang-2019-rmsnorm]。

## 为什么可以省掉 re-centering

LayerNorm 对 $a$ 的共模平移不变：$a_i\mapsto a_i+c$ 时 $\mu\mapsto\mu+c$、$a_i-\mu$ 与 $\sigma$ 均不变，输出不变（由式 2–3 直接得到）[INFERENCE]。这使参数空间出现不可辨识方向：$W\mapsto W+\delta\,\mathbf{1}_n\mathbf{1}_m^{\top}$ 给每个 $a_i$ 加上同一个 $\delta\sum_j x_j$，LayerNorm 的分子把该项整体消掉，输出不变；Table 1 中 LayerNorm 的「权重矩阵 re-centering」列为是，即指此项[^src-zhang-2019-rmsnorm]。被消掉的是 $a$ 的一个标量自由度——沿神经元轴的共模分量，LayerNorm 用一次完整规约把它投影到零，参数中沿 $\mathbf{1}_n\mathbf{1}_m^{\top}$ 的秩一分量随之不可辨识。

RMSNorm 不做该投影：$\mathrm{RMS}(a+c\mathbf{1})\neq\mathrm{RMS}(a)$，共模分量重新对输出可见，权重矩阵与数据的 re-centering 不变性一并丢失（Table 1 的三个 re-centering 列全为否）[^src-zhang-2019-rmsnorm]。这省去了 $\mu$ 这一规约，以及 $\sigma$ 对 $\mu$ 的串行依赖；同一个标量 $\sqrt{\frac{1}{n}\sum_i a_i^{2}}$ 替代两个统计量，也无需「先中心化再平方」。

不变性的前提：

式 6 写作 $\mathrm{RMS}(\alpha x)=\alpha\,\mathrm{RMS}(x)$，严格成立的是 $\mathrm{RMS}(\alpha x)=|\alpha|\,\mathrm{RMS}(x)$；$\alpha<0$ 时 $\bar{a}'=\frac{\alpha}{|\alpha|}\bar{a}$ 整体翻号，式 7 等式只在正比例下完整成立 [INFERENCE]。缩放需作用于整个 $W$，且所有神经元共用同一个 $\delta$；逐行缩放会打破 RMS 线性，Table 1 中 RMSNorm 与 LayerNorm 的「权重向量 re-scaling」同为否[^src-zhang-2019-rmsnorm]。不变性论证不要求分量独立，该假设仅出现在 pRMSNorm：用前 $p\%$ 神经元的均方代替全体，前提是同一层神经元的 summed inputs 近似独立同分布（§5）[^src-zhang-2019-rmsnorm]。

## 梯度

$\mathrm{RMS}$ 为二次型，反向传播梯度式（式 8–10，§4.2）比 LayerNorm 少一项中心化投影。记 $v=\frac{Wx}{\mathrm{RMS}(a)}\odot g+b$ 为式 5 中 $f$ 的入参，$\partial L/\partial v$ 为从 $L$ 传回的梯度：

$$\begin{aligned}
\frac{\partial L}{\partial b} &= \frac{\partial L}{\partial v} && \text{// } [B,n]\to[B,n]\ \text{——偏置梯度直接透传，不带归一化修正}\\
\frac{\partial L}{\partial g} &= \frac{\partial L}{\partial v}\odot\frac{Wx}{\mathrm{RMS}(a)} && \text{// } [B,n]\odot[B,n]\xrightarrow{\text{沿 } B \text{ 维求和}}[n]\ \text{——梯度乘的是归一化后的输入而非原始输入，} g \text{ 的量级因此稳定}\\
\frac{\partial L}{\partial W} &= \sum_{i=1}^{n}x^{\top}\otimes\operatorname{diag}\!\left(g\odot\frac{\partial L}{\partial v}\right)\times R && \text{// 与 } W\in\mathbb{R}^{n\times m} \text{ 同形，} R \text{ 耦合输入与权重；} \otimes\text{、}\times \text{ 的中间维论文未标 } [INFERENCE]
\end{aligned}$$

$$R=\frac{1}{\mathrm{RMS}(a)}\left(I-\frac{(Wx)(Wx)^{\top}}{n\,\mathrm{RMS}(a)^{2}}\right)$$

把 $x$ 或 $W$ 整体乘 $\delta$ 时 $R\mapsto R/\delta$（式 10），代回式 9 可知：$\partial L/\partial W$ 对输入缩放不变，对权重缩放呈负相关，起隐式学习率调节作用，抑制权重矩阵范数膨胀[^src-zhang-2019-rmsnorm]。对比 LayerNorm，两者对 $a$ 的 Jacobian 只差中心化投影项：

$$\frac{\partial\bar{a}}{\partial a}\bigg|_{\text{LayerNorm}}=\frac{g}{\sigma}\left(I-\frac{1}{n}\mathbf{1}\mathbf{1}^{\top}-\frac{(a-\mu)(a-\mu)^{\top}}{n\sigma^{2}}\right),\qquad \frac{\partial\bar{a}}{\partial a}\bigg|_{\text{RMSNorm}}=\frac{g}{\mathrm{RMS}(a)}\left(I-\frac{aa^{\top}}{n\,\mathrm{RMS}(a)^{2}}\right)$$

RMSNorm 缺少 $\frac{1}{n}\mathbf{1}\mathbf{1}^{\top}$ 项，$a-\mu$ 退回 $a$ [INFERENCE]。论文只给出 RMSNorm 的式 8–10，未并列给出 LayerNorm 的对应梯度。

## 证据

来源 arXiv:1910.07467v1（NeurIPS 2019）。加速比均为 RMSNorm 相对 LayerNorm 的运行时间下降；测速在单张 TITAN X (Pascal) 上取 3 次平均，Table 4 的 Transformer 用 Tesla V100，CIFAR-10 用 RTX 2080 Ti。任务为 WMT14 En-De 机器翻译（GRU 的 RNNSearch 与 base Transformer）、CNN/Daily Mail 阅读理解、MS COCO 图文检索、CIFAR-10 分类。

| 设置 | Baseline | LayerNorm | RMSNorm | 加速 |
|---|---|---|---|---|
| RNNSearch，TF/Nematus：Test14 / Test17 BLEU，s/1k 步（Table 2） | 21.7 / 23.4，399±3.40 | 22.6 / 23.6，665±32.5 | 22.4 / 23.7，501±11.8 | 24.7% |
| RNNSearch，Theano/Nematus（Table 3） | 21.8 / 22.9，596±20.8 | 22.3 / 23.8，988±1.10 | 22.5 / 23.2，652±24.1 | 34.0% |
| RNNSearch，PyTorch（Table 3） | 22.7 / 24.7，427±6.50 | 23.2 / 24.3，857±17.2 | 22.9 / 24.5，763±16.2 | 11.0% |
| Transformer base，300K 步（Table 4） | 训练失败 | 26.6 / 27.7，248±1.31 | 26.8 / 27.7，231±0.04 | 6.9% |
| Attentive reader，s/0.1k 步（Table 6） | 315±6.30 | 392±5.70 | 333±5.20 | 15.1% |
| OE 图文检索，s/0.1k 步（Table 8） | 2.11±0.047 | 12.02±0.191 | 7.12±0.207 | 40.8% |
| ConvPool-CNN-C，测试错误率 / s/epoch（Table 10） | 8.96%，21±0.0 | 10.49%，39±0.4 | 8.83%，31±0.5 | 20.5% |

RNNSearch 上 LayerNorm 与 RMSNorm 均减少约一半收敛步数且测试 BLEU 相当；Transformer 不加归一化无法训练（Table 4 中 Baseline 记「-」、BLEU 0），RMSNorm 与 LayerNorm 处于同水平[^src-zhang-2019-rmsnorm]。隐藏状态统计（Table 5，RNNSearch 解码器 GRU 的 hidden-to-hidden 映射，token 位置 1–4 与全位置平均）显示，Baseline 出现明显漂移（位置 1 为 −2.60 / 7.35，全位置平均 −1.60 / 3.04）；LayerNorm 位置 1 为 −0.43 / 1.19、全位置平均 −0.51 / 1.51；RMSNorm 位置 1 为 −0.40 / 1.27、全位置平均 −0.73 / 1.50；RMSNorm 未归一化的均值也比 Baseline 更稳定，直接支持「re-centering 非必需」[^src-zhang-2019-rmsnorm]。图文检索（Table 7，COCO 5 个测试集平均）中，RMSNorm 字幕检索 R@1 / R@5 / R@10 = 48.7 / 79.7 / 89.5、mean rank 5.3，图像检索 39.0 / 74.8 / 86.3、mean rank 7.5；LayerNorm 对应为 47.9 / 79.5 / 89.2（5.3）与 38.4 / 74.6 / 86.7（7.5）[^src-zhang-2019-rmsnorm]。CIFAR-10 上，LayerNorm 把测试错误率从 8.96% 抬高到 10.49%，RMSNorm 为 8.83%，低于 Baseline 的 8.96%，BatchNorm 8.25% 与 WeightNorm 8.28% 仍更好；§6.4 正文称 RMSNorm「surpassing the Baseline by 0.013%」，按 Table 10 的 8.83% 与 8.96% 计算为 0.13 个百分点，正文与表格口径不一致[^src-zhang-2019-rmsnorm]。初始化扰动（Figure 4）显示，权重初始化中心移到 0.2 时 LayerNorm 训练明显不稳定，RMSNorm 仍收敛，两者均不如原始初始化[^src-zhang-2019-rmsnorm]。pRMSNorm 在 $p=6.25\%$ 时 RNNSearch BLEU 与 RMSNorm 同水平（Table 2/3）；$p$ 从 10% 扫到 100% 对最终 BLEU 影响很小（Figure 3）；相对 LayerNorm 的加速在 OE 上达 63.9%（Table 8）、attentive reader 上达 15.8%（Table 6）、CIFAR-10 上再提速 2.6%（§6.4；Table 10 记 23.1% 对 20.5%），三处差异来自实现而非计算量[^src-zhang-2019-rmsnorm]。L2-Norm（将 RMS 换成欧氏范数，两者只差 $\sqrt{n}$ 因子）在 RNNSearch 上未带来增益：Test14 / Test17 为 20.7 / 22.0，低于 Baseline 的 21.7 / 23.4（Table 2）；论文将其归因于按输入向量长度缩放球面，使归一化对不同大小的向量更稳健（§4）[^src-zhang-2019-rmsnorm]。

## 适用范围与边界

统计量只取自样本自身的 summed inputs、不跨训练样本，故不依赖 batch 组成，可在变长序列上逐样本归一化（§1）[^src-zhang-2019-rmsnorm]。加速比取决于框架、硬件与其余组件的相对开销：同为 RNNSearch，Theano 34.0%、TF 24.7%、PyTorch 11.0%；Transformer 因串行归一化操作少，RMSNorm 仅 6.9%、pRMSNorm 9.3%（§6.1、§7）[^src-zhang-2019-rmsnorm]。实验覆盖 GRU RNN、自注意力 Transformer、LSTM attentive reader、GRU order-embedding、CNN 五类结构与四类任务；未覆盖超大模型、低精度训练与 kernel 融合，7%~64% 的效率收益与这些手段正交（§7）[^src-zhang-2019-rmsnorm]。图像分类上归一化普遍不如 BatchNorm / WeightNorm，RMSNorm 仅做到不掉点，CIFAR-10 是唯一让 LayerNorm 掉点的任务（§6.4）[^src-zhang-2019-rmsnorm]。pRMSNorm 理论计算更少但实测未稳定提速，归因于张量切片实现（§6.1、§7）[^src-zhang-2019-rmsnorm]。有偏估计与梯度稳定性只在 $p\geq6.25\%$ 的经验范围内成立，小 $k$ 时梯度爆炸，未给理论界（§5）[^src-zhang-2019-rmsnorm]。

## 相关页面

[[attention-logit-explosion]] 将 RMSNorm 作为不直接约束 Q/K 范数的稳定化选项；[[guided-layer-normalization]] 同属 LayerNorm 族的条件化改造，改的是仿射参数而非统计量；[[instance-normalization]] 属于按样本统计量去 batch 依赖的另一路线；[[transformer-architecture]] 涉及 post-LN 摆放位置，RMSNorm 可原位替换其中的 LayerNorm；[[time-moe]]、[[zeus]]、[[opencity]] 等下游模型把 RMSNorm 作为默认归一化。

## 引用

[^src-zhang-2019-rmsnorm]: [[source-zhang-2019-rmsnorm]]
