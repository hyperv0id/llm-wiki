---
title: "扩散模型"
type: concept
tags:
  - generative-model
  - diffusion
  - vae
  - score-based
created: 2026-04-28
last_updated: 2026-11-24
source_count: 30
confidence: high
status: active
---

# 扩散模型

**扩散模型**是一类生成模型，通过逐步向数据添加噪声（前向过程），然后学习逆转该过程（反向/去噪过程），从而从纯噪声中生成新样本。[^src-understanding-diffusion-models]

## 与变分自编码器的关系

扩散模型可以理解为一种特殊的**马尔可夫层次变分自编码器（Markovian HVAE）**。具体而言，变分扩散模型（VDM）是 Markovian HVAE 在以下三个限制条件下的特例：[^src-understanding-diffusion-models]

1. **潜变量维度等于数据维度**：所有隐变量 $z_t$ 与数据 $x$ 具有相同的维度，不进行维度压缩。
2. **编码器是固定的线性高斯变换**：前向过程不是学习得到的，而是预先定义的高斯转移核，没有可训练参数。
3. **最终潜变量是标准高斯分布**：$p(z_T) = \mathcal{N}(0, I)$，即经过足够多的扩散步后，数据分布被完全破坏为纯噪声。

## 前向过程

前向过程是一个固定的马尔可夫链，逐步向数据添加高斯噪声：[^src-understanding-diffusion-models]

$$
q(x_t | x_{t-1}) = \mathcal{N}\left(\sqrt{\alpha_t}\, x_{t-1},\; (1 - \alpha_t) I\right)
$$

其中 $\alpha_t \in (0, 1)$ 是噪声调度参数。通过[[reparameterization-trick|重参数化技巧]]，可以直接从 $x_0$ 一步采样到任意时间步 $t$——将随机采样 $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)$ 重排为确定性变换加固定噪声 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon$，使前向过程每步可导，这是扩散模型端到端训练的根本保证[^src-bluuuuue-reparameterization-trick]：

$$
q(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right), \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$x_T$ 近似服从标准高斯分布。[^src-understanding-diffusion-models]

## 反向过程

反向过程学习逆转前向加噪过程。模型 $p_\theta(x_{t-1} | x_t)$ 被参数化为高斯分布，其均值由神经网络预测：[^src-understanding-diffusion-models]

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right)
$$

训练目标是最小化反向过程与真实后验 $q(x_{t-1} | x_t, x_0)$ 之间的 KL 散度，该后验在给定 $x_0$ 的条件下具有闭式解。[^src-understanding-diffusion-models]

## 三种等价的训练目标

扩散模型的训练目标有三种等价的参数化形式，它们对应不同的预测任务：[^src-understanding-diffusion-models]

1. **预测 $x_0$**：直接训练网络从噪声图像 $x_t$ 中恢复原始数据 $x_0$。
2. **预测噪声 $\varepsilon$**：训练网络预测添加到 $x_t$ 中的噪声分量。这是 DDPM 中最常用的形式，对应的简化损失函数为：
   $$
   L_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]
   $$
3. **预测得分 $\nabla \log p(x_t)$**：训练网络估计对数概率密度的梯度。

这三种形式通过重参数化在数学上相互等价（给定一种预测可推导出另外两种），但 Li & He (2025) 指出，**在流形假设下，预测目标的根本性质不同**：x-prediction 的输出位于低维数据流形上（on-manifold），而 ε-和 v-prediction 的输出散布在整个高维空间中（off-manifold）。当 patch/输入维度接近或超过网络容量时，只有 x-prediction 能正常工作，ε-/v-prediction 会灾难性失败。[^src-back-to-basics-let-denoising-generative-models-denoise]

x-prediction 最早可追溯到 DDPM 的原始代码[^src-back-to-basics-let-denoising-generative-models-denoise]，但长期未被作为主要预测目标。Li & He 提出的 [[jit|JiT]]（Just image Transformers）系统论证了 x-prediction 在高维空间的必要性，使用标准 ViT + 大 patch + x-prediction 在像素空间实现了有竞争力的生成质量（JiT-G/16: FID 1.82 @ ImageNet 256）。详见 [[x-prediction]]。

## 与基于得分的模型的联系

扩散模型与**基于得分的生成模型**之间存在深刻联系。通过 Tweedie 公式，给定 $x_t$ 时 $x_0$ 的条件期望可以表示为：[^src-understanding-diffusion-models]

$$
\mathbb{E}[x_0 | x_t] = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

这表明预测噪声 $\varepsilon_\theta(x_t, t)$ 等价于估计得分函数 $\nabla_{x_t} \log p(x_t)$，两者之间仅差一个常数因子。因此，训练一个去噪扩散模型本质上等价于在多个噪声水平上进行得分匹配。[^src-understanding-diffusion-models]

## 采样过程

采样通过迭代去噪进行，从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步应用反向转移核：[^src-understanding-diffusion-models]

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

这一过程可以看作是**朗之万动力学**的离散化模拟——每一步都沿着得分函数的估计方向移动，同时注入适量噪声以保持采样多样性。[^src-understanding-diffusion-models]

## 条件生成

扩散模型支持两种主要的条件生成策略：[^src-understanding-diffusion-models]

- **分类器引导（Classifier Guidance）**：利用一个预训练的分类器 $p(y | x_t)$ 的梯度来引导采样过程，使生成结果符合特定类别 $y$。采样时，在得分估计中加入分类器的对数梯度项。
- **无分类器引导（Classifier-Free Guidance）**：同时训练条件模型 $\varepsilon_\theta(x_t, t, y)$ 和无条件模型 $\varepsilon_\theta(x_t, t, \emptyset)$，在采样时对两者进行插值：
  $$
  \tilde{\varepsilon}_\theta = \varepsilon_\theta(x_t, t, \emptyset) + w \left( \varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \emptyset) \right)
  $$
  其中 $w \geq 0$ 控制引导强度。无分类器引导是目前最广泛使用的条件生成方法。[^src-understanding-diffusion-models]

## SDE 统一视角

把离散步数推到连续时间，前向加噪与反向去噪都写成随机微分方程：[^src-chan-2025-diffusion-tutorial]

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

离散侧的两条主线在此汇合：DDPM 在固定方差的高斯噪声链上学习去噪网络，训练目标等价于 ELBO；SMLD 在多尺度高斯扰动上用朗之万动力学采样，训练目标为多尺度得分匹配。两者分别对应方差保持（VP）与方差爆炸（VE）SDE，逆向过程是反向时间 SDE，其边际密度的演化由福克-普朗克方程描述。[^src-chan-2025-diffusion-tutorial]

[[schrodinger-bridge|Schrödinger bridge]] 把这一族过程放进路径空间的变分框架：在两端边缘分布的约束下，最小化相对 reference path measure 的 KL 散度。扩散模型是前向过程固定（OU/VE/VP）的特例，SB 用 [[entropic-optimal-transport|熵正则最优传输]]把同一范式推广到可学习的路径。[^src-schrodinger-bridges-generative-modeling]

## 非高斯扩散（广义扩散模型）

Cold Diffusion（Bansal et al., 2022）把前向过程从高斯加噪换成任意退化算子 $\mathcal{D}$，反向过程学习逆转该退化，构成广义扩散模型框架[^src-dyffusion]。按退化算子的性质，这条支线分为两类：

- **时序退化**：[[dyffusion|DYffusion]] 用时间插值作为前向、时序外推作为反向，把扩散步与物理时间步直接耦合；[[cold-sampling|Cold Sampling]] 是 DDIM 在广义框架上的推广，等价于用 Euler 法求解隐式动力系统的 ODE[^src-dyffusion]。[[armd|ARMD]] 更进一步，以"演化—去演化"取代"加噪—去噪"，用[[sliding-window-diffusion|滑动窗口]]产生的确定性中间态替代白高斯噪声，使采样过程本身即预测过程[^src-armd]。
- **确定性桥**：扩散桥把前向过程的终点钉在确定性目标而非纯高斯噪声，[[brownian-bridge-diffusion|布朗桥扩散]]是代表；[[s2dbm|S²DBM]] 据此把后验方差缩为 0，得到无噪声确定性采样，用于时间序列点预测[^src-s2dbm]。

## 前向过程的归纳偏置塑造

标准扩散模型的前向过程使用各向同性白噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$，对所有频率成分均匀施加噪声：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}$$

Jiralerspong 等人（2025）提出[[frequency-based-noise-control|频域噪声控制]]方法，将白噪声替换为频域塑形噪声 $\boldsymbol{\epsilon}^{(w)}$：[^src-2502-10236]

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}^{(w)}$$

频域塑形流程为：

$$\boldsymbol{\epsilon} \xrightarrow{\mathcal{F}} \mathbf{N}_{\text{freq}} \xrightarrow{w(\mathbf{f})} \mathbf{N}_{\text{freq}}^{(w)} \xrightarrow{\mathcal{F}^{-1}} \boldsymbol{\epsilon}^{(w)}$$

其中 $\mathbf{N}_{\text{freq}} = \mathbf{N}_{\text{real}} + i\mathbf{N}_{\text{imag}}$ 为复高斯随机场，$w(\mathbf{f})$ 为频域加权函数。由于高斯分布的傅里叶变换仍是高斯，频域加权保持高斯假设不变。

核心洞见：**前向加噪过程中被抹除的信息，恰好是去噪模型有压力去学习的信息**。因此通过选择性强调/抑制/跳过特定频段的噪声，可以引导模型聚焦于数据分布的特定方面。实验在 5 个数据集（MNIST, CIFAR-10, DomainNet-Quickdraw, WikiArt, CelebA）中的 3 个上取得了 FID/KID 的显著改善，并展示了选择性忽略被噪声破坏频段的能力。[^src-2502-10236]

Falck 等人（Microsoft Research, 2025）从 SNR 角度给出另一种频域解释：若 $y_0=Fx_0$ 且 $C_i=\operatorname{Var}[(y_0)_i]$，则 DDPM 在频率 $i$ 的 SNR 为 $s_t^{\mathrm{DDPM}}(i)=\bar\alpha_t C_i/(1-\bar\alpha_t)$；自然图像等数据的 $C_i$ 随频率升高而快速下降，因此高频在前向过程中更早、更快被破坏。[^src-equal-snr] 他们提出 [[equal-snr|EqualSNR]]，令傅里叶噪声协方差满足 $\Sigma_{ii}=cC_i$，使所有频率在同一时间步拥有相同 SNR；该方法在标准图像 FID 上与 DDPM 大体持平，同时明显改善高频谱统计。[^src-equal-snr]

## 局限性

扩散模型存在以下主要局限：[^src-understanding-diffusion-models]

- **采样速度慢**：生成一个样本需要数十到数千步的迭代去噪过程，远慢于 GAN 或 VAE 的单步生成。加速方法包括知识蒸馏、快速 ODE 求解器和一致性模型。
- **无压缩的潜变量表示**：由于潜变量维度等于数据维度，扩散模型不提供有意义的低维潜空间，难以进行语义编辑或插值操作。

## 应用

图像生成、音频合成、分子设计、逆问题求解（去模糊、超分辨率、修复）是扩散模型的主战场[^src-chan-2025-diffusion-tutorial]。时空预测上的两条代表路线：[[tcp-diffusion|TCP-Diffusion]] 用[[adjacent-residual-prediction|相邻残差]]替代绝对降水量作为扩散目标，作者报告在 TIGGE 上 ETS-6 0.412、ECMWF-IFS 0.302[^src-tcp]；[[craft|CRAFT]] 用 DDPM 主干叠加[[geographic-feature-alignment|地理特征对齐]]与[[retrieval-based-condition-augmentation|检索增强条件]]，实现零样本跨城市交通流生成[^src-craft]。

## 关键实现与前沿谱系对照

扩散模型从早期离散马尔可夫链演进至连续时间统一视角与时序专用扩展，核心围绕**前向退化与初始隐态**、**目标参数化与连续路径**以及**采样求解器与引导调度**三条正交设计轴分化。在路径与采样轴上，[[ncsn|NCSN]] 与 [[ddpm|DDPM]] 由 [[score-based-sde|Score-Based SDE]] 统一为 VP/VE 随机微分方程及概率流 ODE[^src-sde]；[[ddim|DDIM]]（PF-ODE 一阶 Euler 离散化）、[[dpm-solver|DPM-Solver]]、[[edm|EDM]]（Heun 二阶与对数正态噪声）、[[flow-matching|Flow Matching]]（OT 路径推广）、[[instaflow|InstaFlow]]（reflow 蒸馏）、[[consistency-models|Consistency Models]] 与 [[shortcut-models|Shortcut Models]] 逐步将几十至上千步推理压缩到少步乃至单步[^src-ddpm][^src-ddim][^src-ddim-ode-spaces-ac-cn][^src-dpm-solver][^src-edm][^src-flow-matching][^src-instaflow][^src-consistency-models][^src-shortcut-models]。在主干与参数化轴上，[[dit|DiT]] 以 ViT + adaLN-Zero 取代 U-Net，[[jit|JiT]] 则论证流形假设下大 patch 像素空间须用 `x-prediction` 替代噪声预测[^src-dit][^src-back-to-basics-let-denoising-generative-models-denoise]。

面向时间序列与物理动力学时，各分支进一步在噪声分布与条件控制上特化：[[erdm|ERDM]] 将 EDM 与渐进噪声滚动扩散结合用于动力学概率预测[^src-erdm]；[[nsdiff|NsDiff]] 引入[[location-scale-noise-model|位置—尺度噪声模型（LSNM）]]处理非平稳时变方差[^src-nsdiff]；[[stats|StaTS]] 联合学习[[spectral-trajectory-scheduler|频谱轨迹调度（STS）]]与[[frequency-guided-denoiser|频率引导去噪（FGD）]][^src-stats]；[[fence|FENCE]] 通过[[feedback-diffusion-guidance|反馈扩散引导]]按后验似然动态调节 CFG 尺度[^src-fence]；[[timedit|TimeDiT]] 以 DiT 统一多任务掩码并在推理期注入 PDE 约束朗之万采样，[[swift|Swift]] 以一阶一致性模型将天气预报 NfE 降至 $1$[^src-timedit][^src-swift]；而 [[source-tadiff|TADiff]] 沿用 [[ddim|DDIM]] 确定性概率流，将反向采样起点由标准高斯噪声替换为剥离历史文本条件 $c_h$ 后的前向归因隐态 $x_{h,T}$（增加一条前向链使采样计算约翻倍），在无显式 SCM 下实现自然语言反事实预测[^src-tadiff]。

| 代表路线 | 前向退化 / 初始隐态 | 路径参数化 / 主干 | 采样机制与步数 | 适用边界与主要代价 |
|---|---|---|---|---|
| **[[ddpm\|DDPM]] / [[ncsn\|NCSN]]** | 离散 VP 加噪 / 噪声扰动 $\mathcal{N}(0,I)$ | $\varepsilon$ 或得分匹配 / U-Net | 随机马尔可夫链 / 退火朗之万（$10^2\text{–}10^3$ 步） | 高质量生成但采样慢[^src-ddpm] |
| **[[score-based-sde\|Score-Based SDE]] / [[ddim\|DDIM]]** | 连续 VP/VE SDE，起点 $\mathcal{N}(0,I)$ | 得分 / $\varepsilon$ 预测与 PF-ODE | PC 采样或 PF-ODE 一阶 Euler（10–50 步） | 实现确定性加速与可逆编码[^src-sde][^src-ddim][^src-ddim-ode-spaces-ac-cn] |
| **[[dpm-solver\|DPM-Solver]] / [[edm\|EDM]] / [[erdm\|ERDM]]** | 高斯或渐进滚动噪声调度 | 半线性 ODE / 预处理 $x$ 混合（3D 主干） | 解析指数积分或 Heun 二阶（约 10–39 步） | 兼顾精度与动力学长程采样[^src-dpm-solver][^src-edm][^src-erdm] |
| **[[flow-matching\|Flow Matching]] / [[instaflow\|InstaFlow]] / [[consistency-models\|CM]] / [[shortcut-models\|Shortcut]]** | OT 条件直线流或 PF-ODE 轨迹起点 | 速度场 / 轨迹映射 / 步长条件化 | 1–4 步确定性生成（reflow 或自洽训练） | 极速采样，需蒸馏或自洽目标[^src-flow-matching][^src-instaflow][^src-consistency-models][^src-shortcut-models] |
| **[[dit\|DiT]] / [[jit\|JiT]] / [[timedit\|TimeDiT]]** | 高斯噪声或统一掩码输入 | ViT（adaLN-Zero / x-pred + PDE 引导） | 扩散或物理知情朗之万采样 | 扩展性佳（ImageNet 256 FID 1.82–2.27）[^src-dit][^src-back-to-basics-let-denoising-generative-models-denoise][^src-timedit] |
| **[[nsdiff\|NsDiff]] / [[stats\|StaTS]] / [[fence\|FENCE]] / [[swift\|Swift]]** | LSNM 时变方差或自适应频谱调度 | 非平稳噪声 / FGD 去噪 / 动态 CFG / CM | 动态引导或 1-NFE 自回归一致性 | 适配非平稳、弱条件与高效气象预报[^src-nsdiff][^src-stats][^src-fence][^src-swift] |
| **[[source-tadiff\|TADiff]]** | 剥离历史文本 $c_h$ 的前向 DDIM 归因隐态 $x_{h,T}$ | DDIM 条件去噪 + DTTC 对比微调 | 双向确定性概率流（前向归因 + 反向去噪） | 支持自由文本反事实但采样量约翻倍、无 SCM 可识别性[^src-tadiff] |

## 相关概念

- [[reparameterization-trick]] — 重参数化技巧，前向加噪过程的可微性保证
- [[frequency-based-noise-control]] — 频域噪声控制，通过操控噪声频谱塑造归纳偏置
- [[equal-snr]] — 通过 $\Sigma_{ii}=cC_i$ 实现所有频率等 SNR 加噪
- [[frequency-hierarchy-in-diffusion]] — DDPM 的低频到高频隐式生成层级
- [[frequency-diffusion]] — 频域扩散，频域噪声控制的具体技术实现
- [[inductive-bias-shaping]] — 归纳偏置塑造，显式引导模型学习特定方面
- [[freqflow]] — FreqFlow，频率感知流匹配，通过双分支架构显式建模频率成分（2026）
- [[snr-t-bias]] — SNR-t Bias，扩散模型推理阶段的信噪比-时间步错配偏置（CVPR 2026）
- [[dcw]] — DCW，小波域差分校正方法，缓解 SNR-t Bias 的无需训练方法
- [[frequency-aware-conditioning]] — 频率感知条件化概念
- [[dyffusion|DYffusion]] — 前向换成时间插值、反向换成时序外推的广义扩散特例
- [[armd|ARMD]] — 滑动窗口确定性中间态替代白噪声的"演化—去演化"预测范式
- [[s2dbm|S²DBM]] — 布朗桥扩散的后验方差归零特例，确定性时序点预测
- [[nsdiff|NsDiff]] — 位置—尺度噪声模型把 DDPM 推广到非平稳时序
- [[stats|StaTS]] — 联合学习噪声调度与去噪的频谱轨迹框架
- [[timedit|TimeDiT]] — DiT + 扩散采样的时序基础模型
- [[schrodinger-bridge|Schrödinger Bridge]] — 路径空间变分框架，扩散模型是其前向过程固定的特例
- [[x-prediction]] — x-prediction，扩散模型中直接预测干净数据的参数化方式
- [[jit|JiT]] — JiT (Just image Transformers)，基于 x-prediction 的像素空间扩散模型
- [[dit]] — DiT，用 Transformer 替换 U-Net 的扩散模型架构
- [[ddim]] — DDIM，概率流 ODE 的一阶 Euler 离散化，确定性加速采样与可逆编码
- [[rectified-flow|Rectified Flow]] — 通过 rectification 学习直线 ODE 轨迹实现少步生成
- [[probability-flow-ode]] — 概率流 ODE，确定性 ODE 采样与精确似然计算
- [[source-diffusion-ts-st-survey|TS/ST 扩散综述]] — 首篇同时覆盖 time series 与 spatio-temporal 的 diffusion 综述（arXiv 2024，投 ACM CSUR），四维分类（模型类别/任务/数据模态/应用领域）汇总 60 个方法，提出 denoiser-as-prior 框架连接生成采样与 Plug-and-Play 信号恢复
- [[source-diffusion-traffic-flow-inference|DP-TFI]] — 把 DDPM 当训练期数据增强器（而非主生成器）的早期城市计算案例：fine-grained urban traffic flow inference，用 relax matrix 放宽 superregion/subregion 流量守恒的硬约束
- [[source-tadiff]] — 继承 DDIM 并将初始高斯噪声替换为条件剥离前向归因隐态的文本反事实预测

## 引用

[^src-ddpm]: [[source-ddpm]]
[^src-sde]: [[source-sde]]
[^src-dpm-solver]: [[source-dpm-solver]]
[^src-consistency-models]: [[source-consistency-models]]
[^src-flow-matching]: [[source-flow-matching]]
[^src-shortcut-models]: [[source-shortcut-models]]
[^src-dit]: [[source-dit]]
[^src-edm]: [[source-edm]]
[^src-erdm]: [[source-erdm]]
[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
[^src-bluuuuue-reparameterization-trick]: [[source-bluuuuue-reparameterization-trick]]
[^src-2502-10236]: [[source-2502-10236]]
[^src-equal-snr]: [[source-equal-snr]]
[^src-back-to-basics-let-denoising-generative-models-denoise]: [[source-back-to-basics-let-denoising-generative-models-denoise]]
[^src-instaflow]: [[source-instaflow]]
[^src-ddim]: [[source-ddim]]
[^src-ddim-ode-spaces-ac-cn]: [[source-ddim-ode-spaces-ac-cn]]
[^src-chan-2025-diffusion-tutorial]: [[source-chan-2025-diffusion-tutorial]]
[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
[^src-dyffusion]: [[source-dyffusion]]
[^src-armd]: [[source-armd]]
[^src-s2dbm]: [[source-s2dbm]]
[^src-tcp]: [[source-tcp]]
[^src-craft]: [[source-craft]]
[^src-nsdiff]: [[source-nsdiff]]
[^src-stats]: [[source-stats]]
[^src-fence]: [[source-fence]]
[^src-timedit]: [[source-timedit]]
[^src-swift]: [[source-swift]]
[^src-tadiff]: [[source-tadiff]]
