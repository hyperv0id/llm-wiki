---
title: "EqualSNR"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - signal-to-noise-ratio
  - noise-schedule
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# EqualSNR

**EqualSNR** 是 Falck 等人（Microsoft Research, 2025）提出的傅里叶空间扩散前向过程：它把每个频率的噪声方差设置为与该频率的数据方差成比例，使所有频率在同一时间步具有相同信噪比（SNR）。[^src-equal-snr]

这篇论文的核心不是“换一个噪声 schedule”这么简单，而是把标准 [[ddpm|DDPM]] 的前向过程改写到傅里叶空间后指出：DDPM 白噪声在**噪声方差**上对所有频率相同，但在**信号相对噪声比例**上并不公平，因为自然图像、视频、音频、蛋白质密度图等数据的傅里叶方差随频率快速衰减。[^src-equal-snr]

## 1. DDPM 的傅里叶空间等价形式

标准 DDPM 的一步边缘采样可写为：[^src-equal-snr]

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

令 $F$ 为离散傅里叶变换，$y_t=Fx_t$，$y_0=Fx_0$，则由线性性得到：[^src-equal-snr]

$$
\begin{aligned}
y_t
&= F x_t \\
&= F\left(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon\right) \\
&= \sqrt{\bar\alpha_t}\,F x_0 + \sqrt{1-\bar\alpha_t}\,F\epsilon \\
&= \sqrt{\bar\alpha_t}\,y_0 + \sqrt{1-\bar\alpha_t}\,n,
\end{aligned}
$$

其中 $n=F\epsilon$ 仍是复高斯白噪声，因为单位化傅里叶变换保持各向同性高斯的协方差结构。[^src-equal-snr] 因此 DDPM 可以完全等价地被看作傅里叶空间里的加噪过程，只是信号方差在不同频率上并不相等。[^src-equal-snr]

## 2. SNR 定义与 DDPM 的频率不均匀性

论文把复数随机变量的方差定义为：[^src-equal-snr]

$$
\operatorname{Var}(z)=\operatorname{Var}(\operatorname{Re}z)+\operatorname{Var}(\operatorname{Im}z).
$$

若测量过程为 $f(s,n)=s+n$，则 SNR 定义为：[^src-equal-snr]

$$
\operatorname{SNR}(f)=\frac{\operatorname{Var}(s)}{\operatorname{Var}(n)}.
$$

设第 $i$ 个傅里叶频率的数据方差为：[^src-equal-snr]

$$
C_i = \operatorname{Var}\big((y_0)_i\big),
$$

则 DDPM 在时间步 $t$、频率 $i$ 上的 SNR 为：[^src-equal-snr]

$$
s^{\mathrm{DDPM}}_t(i)
= \frac{\bar\alpha_t C_i}{1-\bar\alpha_t}.
$$

这条公式是论文的第一块基石：如果数据满足傅里叶功率律，即 $C_i$ 随频率升高而快速下降，那么即使 DDPM 对所有频率添加同方差白噪声，高频的 $s^{\mathrm{DDPM}}_t(i)$ 也会在所有时间步上显著低于低频。[^src-equal-snr]

直观说，DDPM 不是“所有频率等速率变脏”，而是“高频相对于自身信号更快变脏”。[^src-equal-snr] 反向过程学习镜像前向过程，因此 DDPM 通常先生成低频结构，再在最后补高频细节，形成 [[frequency-hierarchy-in-diffusion|频率层级]]。[^src-equal-snr]

## 3. 为什么高频更快加噪会破坏反向高斯假设

DDPM 的随机反向过程通常假设：[^src-equal-snr]

$$
p_\theta(y_{t-1}\mid y_t)
$$

可以用高斯分布近似真实后验：[^src-equal-snr]

$$
q(y_{t-1}\mid y_t).
$$

但由 Bayes 公式：[^src-equal-snr]

$$
q(y_{t-1}\mid y_t)=\frac{q(y_t\mid y_{t-1})q(y_{t-1})}{q(y_t)}.
$$

如果 $q(y_t\mid y_{t-1})$ 的噪声方差相对于 $q(y_{t-1})$ 和 $q(y_t)$ 的信号方差过大，那么边缘比值 $q(y_{t-1})/q(y_t)$ 的非高斯形状会更明显地传到后验中。[^src-equal-snr] 对高频而言，$C_i$ 很小而 DDPM 噪声方差不随频率降低，所以这种“噪声相对过大”的情况最容易出现。[^src-equal-snr]

论文给出一个一维反例：令前一时刻分布是两个窄高斯的混合，[^src-equal-snr]

$$
D_0 = \frac12\mathcal{N}(-1,\delta^2)+\frac12\mathcal{N}(1,\delta^2),
$$

并令：[^src-equal-snr]

$$
x_{t-1}\sim D_0,
\qquad
\epsilon\sim\mathcal{N}(0,4),
\qquad
x_t=x_{t-1}+\epsilon.
$$

则对足够小的 $\delta,\tau$，以至少 $1-\tau$ 的概率，真实反向分布到任意单一高斯都有正的 total variation 距离下界：[^src-equal-snr]

$$
\inf_{\mu,\sigma>0}D_{\mathrm{TV}}\left(q(x_{t-1}\mid x_t),\mathcal{N}(\mu,\sigma^2)\right)
\ge \Omega(\tau^{18}).
$$

这个反例说明：即使每一步前向转移是高斯，反向条件分布也未必能由单一高斯任意精确近似；当高频被相对过强地加噪时，DDPM 的高斯反向假设更容易失效。[^src-equal-snr]

## 4. EqualSNR 的定义

论文把更一般的傅里叶空间前向过程写成：[^src-equal-snr]

$$
y_t
= \sqrt{\bar\alpha_t}\,y_0
+ \sqrt{1-\bar\alpha_t}\,\epsilon_\Sigma,
\qquad
\epsilon_\Sigma\sim\mathcal{CN}(0,\Sigma).
$$

此时第 $i$ 个频率的 SNR 为：[^src-equal-snr]

$$
s_t(i)
=\frac{\bar\alpha_t C_i}{(1-\bar\alpha_t)\Sigma_{ii}}.
$$

要让所有频率在同一时间步具有相同 SNR，需要存在常数 $c>0$ 使：[^src-equal-snr]

$$
\Sigma_{ii}=cC_i.
$$

当 $c=1$ 时，噪声协方差的对角项直接匹配数据在傅里叶频率上的方差，论文称这种版本是坐标级 variance-preserving。[^src-equal-snr] 这就是 EqualSNR 的核心：它不让白噪声在所有频率上“绝对公平”，而是让噪声相对于各频率信号方差“相对公平”。[^src-equal-snr]

## 5. 与 DDPM / FlippedSNR 的统一形式

在同一个公式下，DDPM、EqualSNR、FlippedSNR 可以写成三种 $\Sigma$ 选择：[^src-equal-snr]

| 前向过程 | SNR 目标 | 噪声协方差条件 |
|---|---|---|
| DDPM | 高频因 $C_i$ 小而更快降 SNR | $\Sigma_{ii}=1$ |
| EqualSNR | 所有频率同时间步同 SNR | $\Sigma_{ii}=cC_i$ |
| FlippedSNR | 频率层级相对 DDPM 翻转 | $\Sigma_{ii}=C_i/C_{d-i}$ |

FlippedSNR 的目标是让反向过程先生成高频、再生成低频；作者尝试多种 beta schedule、SNR 起止值和 $T\in\{1000,5000,10000\}$ 的离散步数，但没有成功训练出合理样本。[^src-equal-snr] 这不是严格不可能性证明，但它提示“低频到高频”的层级可能在优化或表示上有实际作用。[^src-equal-snr]

## 6. 公平比较所需的 SNR 校准

论文强调，比较不同前向过程时不能只换 $\Sigma$，还要让平均 SNR 在每个时间步可比。[^src-equal-snr] 设 DDPM 与 EqualSNR 的平均 SNR 分别为 $A_{\mathrm{ddpm}}(t)$ 和 $A_{\mathrm{eq}}(t)$，若固定 DDPM 的 $\bar\alpha^{\mathrm{ddpm}}_t$，需要求 EqualSNR 的 $\bar\alpha^{\mathrm{eq}}_t$ 使：[^src-equal-snr]

$$
A_{\mathrm{eq}}(t)=A_{\mathrm{ddpm}}(t).
$$

根据论文附录推导，校准后的 EqualSNR mixing coefficient 可写为：[^src-equal-snr]

$$
\bar\alpha^{\mathrm{eq}}_t
=
\frac{
\bar\alpha^{\mathrm{ddpm}}_t\left(\frac1d\sum_i C_i\right)
}{
1-\bar\alpha^{\mathrm{ddpm}}_t
+\bar\alpha^{\mathrm{ddpm}}_t\left(\frac1d\sum_i C_i\right)
}.
$$

这个校准确保两种过程在同一时间步平均破坏的信息量相当；否则 EqualSNR 与 DDPM 的 FID 差异可能混入“总噪声量不同”的混杂因素。[^src-equal-snr]

## 7. EqualSNR 的训练目标与 ELBO

EqualSNR 的训练算法保留像素空间 U-Net 输入输出，但把预测结果变换到傅里叶空间计算损失。[^src-equal-snr] 对每个训练样本，先计算：[^src-equal-snr]

$$
y_0=Fx_0,
\qquad
y_t=\sqrt{\bar\alpha_t}y_0+\sqrt{1-\bar\alpha_t}\epsilon_C,
$$

其中 $\epsilon_C$ 是协方差与 $C$ 匹配的复高斯噪声。[^src-equal-snr] 网络在像素空间预测 $\hat{x}_0=f_\theta(F^{-1}y_t,t)$，再变换为：[^src-equal-snr]

$$
\hat{y}_0=F\hat{x}_0.
$$

论文使用的损失为：[^src-equal-snr]

$$
L_t(\theta)=\left\|C^{-1/2}(y_0-\hat{y}_0)\right\|^2.
$$

其中 $C$ 是傅里叶频率的对角方差矩阵。[^src-equal-snr] 这个 $C^{-1/2}$ 权重很关键：它把误差按各频率的自然方差归一化，避免低频因绝对方差大而支配损失。[^src-equal-snr]

论文证明，在反向重建分布采用复高斯形式且协方差与 $C$ 成比例时，最小化上述加权 MSE 等价于最大化标准 ELBO 中的相应 KL 项。[^src-equal-snr] 证明依赖复高斯 KL 的均值项：当两个复高斯的协方差相同或成比例时，KL 关于均值差的部分正比于加权二次型。[^src-equal-snr]

## 8. EqualSNR 的 DDIM 采样形式

论文将 DDIM 更新写到傅里叶空间。[^src-equal-snr] 给定当前 $y_t$，先由网络预测干净样本：[^src-equal-snr]

$$
\hat{y}^{(t)}_0 = (F\circ f_\theta)(F^{-1}(y_t),t).
$$

然后用确定性 DDIM 形式更新：[^src-equal-snr]

$$
y_{t-1}
=\sqrt{\bar\alpha_{t-1}}\hat{y}^{(t)}_0
+\frac{\sqrt{1-\bar\alpha_{t-1}}}{\sqrt{1-\bar\alpha_t}}
\left(y_t-\sqrt{\bar\alpha_t}\hat{y}^{(t)}_0\right).
$$

最终样本通过反傅里叶变换返回像素空间：[^src-equal-snr]

$$
x_0=F^{-1}y_0.
$$

## 9. 实验结论

论文的标准图像 benchmark 包括 CIFAR-10、CelebA 和 LSUN Church。[^src-equal-snr] Clean-FID 结果显示 EqualSNR 在多数设置下与 DDPM 持平，最高分辨率的 LSUN Church 上 EqualSNR calibrated schedule 明显优于对应 DDPM schedule。[^src-equal-snr]

论文更重视高频质量而非单纯 FID。[^src-equal-snr] 在 CIFAR-10 上，作者把生成图和真实图的高频谱幅值拟合成两个统计量，再用 logistic regression 做二分类；对 DDPM，top 5%、15%、25% 高频带的平均分类准确率约为 $0.624,0.643,0.654$，且 100 次测试中几乎总能拒绝“真实与生成同分布”的原假设。[^src-equal-snr]

对 EqualSNR，同样分类器在 top 5%、15%、25% 高频带上的平均准确率约为 $0.516,0.521,0.518$，接近随机分类；这说明 EqualSNR 的高频统计更接近真实数据。[^src-equal-snr]

Dots 合成实验进一步把任务设置为“高频是主体”：图像中只有随机分布的少量白点。[^src-equal-snr] EqualSNR 能更接近真实点数和像素强度斜率，而 DDPM 倾向生成过多、强度偏弱的点。[^src-equal-snr]

## 10. 局限与解释边界

EqualSNR 的优势主要体现在高频关键任务和高频统计上，而不是在标准图像 FID 上全面压倒 DDPM。[^src-equal-snr] 论文最大真实图像实验为 $128\times128$，没有覆盖现代 text-to-image 或 video diffusion 的大模型规模。[^src-equal-snr]

FlippedSNR 训练失败说明“先高频、后低频”的生成顺序可能很难学习，但论文没有证明这种顺序在理论上不可行。[^src-equal-snr] 因此更稳妥的结论是：DDPM 的低频到高频层级不是唯一可行结构，因为 EqualSNR 能工作；但完全翻转该层级可能破坏了某种优化上有用的结构。[^src-equal-snr]

## 相关页面

- [[source-equal-snr]] — 论文源摘要
- [[frequency-hierarchy-in-diffusion]] — DDPM 的低频到高频生成层级
- [[frequency-based-noise-control]] — 频域噪声形状控制
- [[inductive-bias-shaping]] — 通过训练过程设计塑造归纳偏置
- [[diffusion-model]] — 扩散模型基础
- [[ddpm]] — 标准 DDPM 前向过程

## 引用

[^src-equal-snr]: [[source-equal-snr]]
