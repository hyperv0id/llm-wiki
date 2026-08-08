---
title: "扩散模型 (Diffusion Models)"
type: concept
tags:
  - generative-models
  - diffusion
  - deep-learning
created: 2026-04-28
last_updated: 2026-08-08
source_count: 12
confidence: high
status: active
---

# 扩散模型

**扩散模型**是一类通过逐步向数据添加噪声再逆转过程来生成新样本的生成模型。核心思想来自非平衡热力学。[^src-chan-2025-diffusion-tutorial]

## 两类主流方法

### DDPM (Denoising Diffusion Probabilistic Models)

- 正向过程：固定方差的高斯噪声逐步添加
- 逆向过程：学习去噪网络
- 训练目标：ELBO 下界，等价于去噪得分匹配

### SMLD (Score Matching Langevin Dynamics)

- 正向过程：多尺度高斯噪声扰动
- 逆向过程：朗之万动力学采样
- 训练目标：多尺度得分匹配

## SDE 统一视角

连续时间下，扩散过程由随机微分方程描述：

$$
dx = f(x, t) dt + g(t) dw
$$

- **DDPM**：对应方差爆炸型 (VE) SDE
- **SMLD**：对应方差保持型 (VP) SDE

逆向过程对应反向时间 SDE，福克-普朗克方程描述概率密度的演化。[^src-chan-2025-diffusion-tutorial]

### Schrödinger Bridge 统一视角

[[schrodinger-bridge|Schrödinger bridge (SB)]] 提供了一个统一的路径空间变分框架，将 diffusion models、score-based models 和 flow matching 统一在同一个原理下：在边缘分布约束下，最小化相对于 reference path measure 的 KL divergence[^src-schrodinger-bridges-generative-modeling]。扩散模型对应固定前向过程（OU/VE/VP）的特殊情况，而 SB 通过 [[entropic-optimal-transport|entropy-regularized optimal transport]] 在路径空间中推广了这一范式。

## 非高斯扩散（广义扩散模型）

Cold Diffusion（Bansal et al., 2022）提出了"广义扩散模型"框架，将前向过程从高斯加噪推广到任意退化算子 $\mathcal{D}$，反向过程学习逆转该退化[^src-dyffusion]。该框架下：

- **[[dyffusion|DYffusion]]** (NeurIPS 2023)：将扩散前向过程替换为时序插值，反向过程替换为时序预测，将扩散步与物理时间步直接耦合，专门面向时空动力学预测[^src-dyffusion]
- **[[cold-sampling|Cold Sampling]]**：DDIM 在广义扩散模型上的推广，对应 Euler 方法求解隐式动力系统的 ODE[^src-dyffusion]

扩散桥（diffusion bridge）类方法把前向过程钉在确定性终点而非纯高斯噪声，[[brownian-bridge-diffusion|布朗桥扩散]]是其代表——[[s2dbm|S²DBM]] 据此把后验方差缩放为 0 即得无噪声确定性采样，用于时间序列点预测[^src-s2dbm]。

扩散范式也被重新诠释用于时间序列预测：[[armd|ARMD]]（AAAI 2025）以"演化—去演化"取代"加噪—去噪"，用[[sliding-window-diffusion|滑动窗口]]产生的确定性中间态替代白高斯噪声，使扩散采样过程本身即预测过程[^src-armd]。

## 应用领域

- 图像生成（DALL-E, Stable Diffusion 的底层技术）
- 音频合成
- 药物分子设计
- 逆问题求解（去模糊、超分辨率、修复）
- 时空动力学预测（DYffusion）[^src-dyffusion]
- **热带气旋降水预测**：[[tcp-diffusion|TCP-Diffusion]] (ICML 2025) 通过 [[adjacent-residual-prediction|ARP]] 机制将预测目标从绝对降水改为相邻残差，以扩散模型直接建模降水变化，论文报告在 TIGGE 对比中超越 ECMWF-IFS（ETS-6 0.412 vs 0.302）[^src-tcp]
- **跨城市交通流生成**：[[craft|CRAFT]] (NeurIPS 2025) 使用 DDPM 主干 + [[geographic-feature-alignment|地理特征对齐]] + [[retrieval-based-condition-augmentation|检索增强条件]]实现零样本跨城市交通流生成，59.7% 超越基线平均值[^src-craft]

## 挑战与未来方向

- 采样速度慢（需要数十到数百步）
- 与物理世界的物理一致性
- 信息取证与深度伪造检测
- **非平稳时序扩散**：[[nsdiff|NsDiff]] (ICML 2025 Spotlight) 将 DDPM 的固定单位方差假设推广为 [[location-scale-noise-model|LSNM]]，使扩散过程能建模时变不确定性，在 9 个真实数据集上取得 SOTA[^src-nsdiff]
- **频谱轨迹调度学习**：[[stats|StaTS]] (arXiv 2026) 联合优化噪声调度与去噪过程，通过 [[spectral-trajectory-scheduler|STS]] 学习数据自适应调度、[[frequency-guided-denoiser|FGD]] 估计调度诱导频谱失真调制去噪强度，在 8 个基准上以极低内存（27 MB）超越 NsDiff 等基线[^src-stats]
- **动态引导机制**：固定 CFG 引导尺度在条件信息不足时会导致漂移到先验分布，[[feedback-diffusion-guidance|反馈扩散引导]]通过后验似然动态调整引导强度来解决此问题[^src-fence]
- **时间序列扩散基础模型**：[[timedit|TimeDiT]] (KDD 2025) 首次将 DiT 与扩散采样统一为时间序列基础模型，通过统一掩码机制支持预测/插补/异常检测/数据生成四大任务，并以物理信息朗之万采样在推理时注入 PDE 约束（免微调）[^src-timedit]
- **一致性模型加速**：[[swift|Swift]] (arXiv 2025) 通过将扩散模型的 PF-ODE 替换为一阶一致性模型，在天气预测中实现 39× 推理加速，NfE 从 39 降至 1，并首次实现 75 天稳定自回归预报[^src-swift]

知识蒸馏和快速 ODE 求解器是加速采样的主要方向。[^src-chan-2025-diffusion-tutorial]

[^src-tcp]: [[source-tcp]]

## 引用

[^src-chan-2025-diffusion-tutorial]: [[source-chan-2025-diffusion-tutorial]]
[^src-dyffusion]: [[source-dyffusion]]
[^src-fence]: [[source-fence]]
[^src-nsdiff]: [[source-nsdiff]]
[^src-stats]: [[source-stats]]
[^src-timedit]: [[source-timedit]]
[^src-craft]: [[source-craft]]
[^src-swift]: [[source-swift]]
[^src-s2dbm]: [[source-s2dbm]]
[^src-armd]: [[source-armd]]
[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
