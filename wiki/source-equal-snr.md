---
title: "A Fourier Space Perspective on Diffusion Models"
type: source-summary
tags:
  - diffusion-models
  - frequency-domain
  - signal-to-noise-ratio
  - microsoft-research
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# A Fourier Space Perspective on Diffusion Models

Falck 等人（Microsoft Research, 2025）从傅里叶空间分析扩散模型的前向过程，核心问题是：标准 [[ddpm|DDPM]] 的白噪声是否真的“公平”破坏所有频率。[^src-equal-snr] 论文指出，图像、视频、音频、Cryo-EM 蛋白质密度图等模态都具有傅里叶功率律，即低频分量方差和幅值远大于高频分量。[^src-equal-snr] 因此 DDPM 虽然给所有频率加入相同方差的白噪声，但高频分量的 SNR 会更早、更快下降。[^src-equal-snr]

论文首先把 DDPM 边缘前向过程：[^src-equal-snr]

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
$$

改写到傅里叶空间：[^src-equal-snr]

$$
y_t=Fx_t=\sqrt{\bar\alpha_t}y_0+\sqrt{1-\bar\alpha_t}F\epsilon.
$$

设 $C_i=\operatorname{Var}[(y_0)_i]$，则 DDPM 在频率 $i$ 的 SNR 为：[^src-equal-snr]

$$
s_t^{\mathrm{DDPM}}(i)=\frac{\bar\alpha_t C_i}{1-\bar\alpha_t}.
$$

该公式解释了 DDPM 的 [[frequency-hierarchy-in-diffusion|频率层级]]：高频在前向过程更快变成低 SNR，反向过程就更晚生成高频细节。[^src-equal-snr]

理论部分进一步说明，高频快速加噪会让 DDPM 反向过程的单一高斯假设更容易失效。[^src-equal-snr] 反向条件分布满足：[^src-equal-snr]

$$
q(y_{t-1}\mid y_t)=\frac{q(y_t\mid y_{t-1})q(y_{t-1})}{q(y_t)}.
$$

当 $q(y_t\mid y_{t-1})$ 的噪声方差相对高频信号方差过大时，$q(y_{t-1})/q(y_t)$ 的非高斯结构会更明显地影响后验。[^src-equal-snr] 论文给出混合高斯反例，证明即使前向转移为高斯，真实反向分布也可能与任何单一高斯保持正的 total variation 距离。[^src-equal-snr] 主文 Fig. 3 和附录 Fig. 9-14 用 CIFAR-10 的 KDE 可视化展示了 DDPM 高频后验比低频更偏离高斯，而 EqualSNR 的低/高频偏离程度更均衡。[^src-equal-snr]

为修正该问题，论文提出 [[equal-snr|EqualSNR]]，把傅里叶空间前向过程写为：[^src-equal-snr]

$$
y_t=\sqrt{\bar\alpha_t}y_0+\sqrt{1-\bar\alpha_t}\epsilon_\Sigma,
\qquad \epsilon_\Sigma\sim\mathcal{CN}(0,\Sigma),
$$

并令第 $i$ 个频率的 SNR 为：[^src-equal-snr]

$$
s_t(i)=\frac{\bar\alpha_t C_i}{(1-\bar\alpha_t)\Sigma_{ii}}.
$$

所有频率等 SNR 当且仅当 $\Sigma_{ii}=cC_i$；当 $c=1$ 时，噪声协方差对角项直接匹配各频率数据方差。[^src-equal-snr] 论文还给出用于公平比较的平均 SNR 校准公式，确保 DDPM 和 EqualSNR 在同一时间步破坏的平均信息量一致。[^src-equal-snr]

训练上，EqualSNR 保留像素空间 U-Net，但在傅里叶空间计算加权 $x_0$ 预测损失：[^src-equal-snr]

$$
L_t(\theta)=\left\|C^{-1/2}(y_0-\hat y_0)\right\|^2.
$$

作者证明该损失对应标准扩散 ELBO 中的 KL 项，因此 EqualSNR 不是仅靠经验调参的启发式方法。[^src-equal-snr]

实验结果分两层。第一层是标准图像质量：在 CIFAR-10、CelebA、LSUN Church 上，EqualSNR 的 Clean-FID 与 DDPM 大体持平，LSUN Church 等更高分辨率场景中 EqualSNR calibrated schedule 有明显优势。[^src-equal-snr] 第二层是高频质量：只用高频谱幅值拟合出的两个统计量训练 logistic regression，能区分 DDPM 生成图和真实图，但对 EqualSNR 生成图接近随机分类。[^src-equal-snr] Dots 合成实验进一步显示，当任务目标主要是稀疏高频结构时，EqualSNR 更能生成正确数量和强度的点。[^src-equal-snr]

论文也讨论了 FlippedSNR：它把 DDPM 的 SNR 频率剖面翻转，迫使反向过程先生成高频、再生成低频。[^src-equal-snr] 作者尝试多种 schedule、SNR 范围和 $T\in\{1000,5000,10000\}$ 的扩散步数均未成功训练。[^src-equal-snr] 这说明低频到高频的生成层级可能具有优化价值，但 EqualSNR 的成功也说明该层级并非标准图像生成的绝对必要条件。[^src-equal-snr]

主要局限是实验规模仍停留在最高 $128\times128$ 的标准图像数据集，没有验证现代大规模图像/视频 diffusion；FlippedSNR 失败只能作为部分证据，不能证明先高频后低频在理论上不可学。[^src-equal-snr]

## 引用

[^src-equal-snr]: [[source-equal-snr]]
