---
title: "扩散模型中的频率层级"
type: concept
tags:
  - diffusion-models
  - frequency-domain
  - signal-to-noise-ratio
  - inductive-bias
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# 扩散模型中的频率层级

**扩散模型中的频率层级**指标准 [[ddpm|DDPM]] 在傅里叶空间中隐式形成的生成顺序：反向过程通常先恢复低频结构，再补全高频细节。[^src-equal-snr]

这个层级不是网络架构显式规定的，而是由两件事共同决定：数据的傅里叶功率律，以及 DDPM 白噪声前向过程对所有频率施加相同方差噪声。[^src-equal-snr]

## 1. 数据频谱：低频方差远大于高频方差

Falck 等人指出，图像、视频、音频、Cryo-EM 蛋白质密度图等扩散模型擅长的模态具有相似的傅里叶统计：低频分量方差和幅值比高频分量大几个数量级。[^src-equal-snr]

用符号表示，令：[^src-equal-snr]

$$
y_0=Fx_0,
\qquad
C_i=\operatorname{Var}\big((y_0)_i\big),
$$

则这些数据大致满足 $C_i$ 随频率升高快速衰减。[^src-equal-snr]

## 2. DDPM 白噪声：绝对均匀，相对不均匀

DDPM 的边缘前向过程为：[^src-equal-snr]

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

变换到傅里叶空间：[^src-equal-snr]

$$
y_t=\sqrt{\bar\alpha_t}y_0+\sqrt{1-\bar\alpha_t}n,
\qquad
n\sim\mathcal{CN}(0,I).
$$

因此第 $i$ 个频率的 SNR 是：[^src-equal-snr]

$$
s_t^{\mathrm{DDPM}}(i)=\frac{\bar\alpha_t C_i}{1-\bar\alpha_t}.
$$

白噪声让所有频率的噪声方差相同，但 $C_i$ 不同，所以高频的 SNR 从一开始就低得多，并且更早进入低 SNR 区域。[^src-equal-snr]

## 3. 前向层级如何变成反向层级

前向过程里，高频信息更早被破坏，低频信息更晚才消失。[^src-equal-snr] 训练出的反向过程会镜像这种破坏顺序：先从噪声中建立低频结构，再在接近采样末端补回高频细节。[^src-equal-snr]

这可以被看作一种近似自回归结构：高频不是独立生成，而是在低频已经确定的条件下生成。[^src-equal-snr] Dieleman 等相关讨论把这种现象称为 diffusion 的 spectral autoregression，Falck 等人的贡献是把它同 SNR 公式、反向高斯假设和高频质量联系起来。[^src-equal-snr]

## 4. 层级的好处与代价

频率层级可能有优化好处，因为低频承载全局形状、布局和颜色，高频承载纹理、边缘和细节。[^src-equal-snr] 如果模型先解决低频，再条件化生成高频，任务分解可能更自然。[^src-equal-snr]

但该层级也会造成高频质量问题。[^src-equal-snr] 当高频信号方差 $C_i$ 很小而前向噪声不相应变小时，真实后验：[^src-equal-snr]

$$
q(y_{t-1}\mid y_t)=\frac{q(y_t\mid y_{t-1})q(y_{t-1})}{q(y_t)}
$$

更容易受到边缘分布非高斯形状影响，从而偏离 DDPM 反向过程使用的单一高斯近似。[^src-equal-snr]

论文在 CIFAR-10 上用 KDE 可视化了这一点：DDPM 的低频后验近似高斯，而高频后验更明显偏离高斯；EqualSNR 下低频和高频的后验偏离程度更接近。[^src-equal-snr]

## 5. EqualSNR：移除层级的消融实验

[[equal-snr|EqualSNR]] 通过令噪声协方差满足：[^src-equal-snr]

$$
\Sigma_{ii}=cC_i
$$

使所有频率在同一时间步拥有相同 SNR：[^src-equal-snr]

$$
s_t(i)=\frac{\bar\alpha_t C_i}{(1-\bar\alpha_t)\Sigma_{ii}}=\frac{\bar\alpha_t}{c(1-\bar\alpha_t)}.
$$

这相当于移除 DDPM 的低频到高频生成层级，让所有频率以相同速率被破坏，也让反向过程学习同时生成各频率。[^src-equal-snr]

EqualSNR 的实验结果说明：DDPM 的层级不是生成图像的绝对必要条件，因为 EqualSNR 在标准图像 FID 上大体持平。[^src-equal-snr] 同时，EqualSNR 显著改善了高频谱统计，因此 DDPM 层级至少会在高频关键任务上带来代价。[^src-equal-snr]

## 6. FlippedSNR：翻转层级为什么困难

FlippedSNR 是更激进的消融：它把 DDPM 的频率 SNR 剖面翻转，使低频先被破坏、高频后被破坏，从而要求反向过程先生成高频、再生成低频。[^src-equal-snr]

形式上，论文给出 FlippedSNR 的条件：[^src-equal-snr]

$$
s^{\mathrm{FLIP}}_t(i)=s^{\mathrm{DDPM}}_t(d-i)
\quad\Longleftrightarrow\quad
\Sigma_{ii}=\frac{C_i}{C_{d-i}}.
$$

作者尝试了 cosine、linear 和其他 SNR schedule，调整起止 SNR，并把扩散步数扩展到 $T\in\{1000,5000,10000\}$，但 FlippedSNR 仍未成功训练。[^src-equal-snr]

这个失败说明“完全先高频后低频”的顺序可能不利于优化，但它不能证明这种顺序理论上不可学习。[^src-equal-snr] 更稳妥的解释是：低频到高频层级不是必要条件，但可能是一种有用的生成结构；EqualSNR 移除层级仍可工作，FlippedSNR 完全反转层级则很难训练。[^src-equal-snr]

## 7. 与相关概念的关系

- [[equal-snr]] — 直接控制每频率 SNR，移除 DDPM 频率层级
- [[frequency-based-noise-control]] — 更一般地操控频域噪声形状来塑造扩散模型偏置
- [[inductive-bias-shaping]] — 前向信息破坏方式决定模型学习压力
- [[snr-t-bias]] — 另一个 SNR 视角的扩散模型偏置，但关注推理阶段实际 SNR 与时间步错配
- [[diffusion-model]] — 扩散模型基础

## 引用

[^src-equal-snr]: [[source-equal-snr]]
