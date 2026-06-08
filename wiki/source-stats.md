---
title: "StaTS — Spectral Trajectory Schedule Learning for Adaptive Time Series Forecasting with Frequency Guided Denoiser"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - noise-schedule
  - frequency-domain
  - arxiv-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: StaTS

**作者**: Anonymous (arXiv 2603.00037, submitted concurrently with PAFM to ICML)
**发表**: arXiv 2603
**领域**: 扩散概率时间序列预测
**代码**: https://github.com/zjt-gpu/StaTS

## 核心论点

StaTS 提出一个联合优化噪声调度和去噪过程的扩散预测框架，由两部分构成：**Spectral Trajectory Scheduler (STS)** 学习数据自适应的噪声调度，**Frequency Guided Denoiser (FGD)** 调度感知频谱损伤以调制去噪强度[^src-stats]。

核心洞察：固定噪声调度（线性/余弦/二次）产生与数据集特征不对齐的频谱衰减轨迹，导致中间加噪状态难以去噪，损害预测质量和不确定性校准[^src-stats]。STS 通过频域正则化学习自适应调度，FGD 利用调度诱导的频谱失真信息指导异构恢复[^src-stats]。

## 方法

### STS (Spectral Trajectory Scheduler)

STS 将噪声调度 θ 作为可学习参数，在两阶段交替优化中与 FGD 联合训练[^src-stats]：

**目标函数**：

$$\mathcal{R}(\beta) = \mathcal{L}_{\text{pred}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{init}} \mathcal{L}_{\text{init}} + \lambda_{\text{end}} \mathcal{L}_{\text{end}} + \lambda_{\text{bar}} \mathcal{L}_{\text{bar}} + \lambda_{\text{flatness}} \mathcal{L}_{\text{flatness}}$$

其中 $\mathcal{L}_{\text{pred}}$ 是预测导向目标，$\mathcal{L}_{\text{smooth}}$ 约束调度平滑性，$\mathcal{L}_{\text{init}}/\mathcal{L}_{\text{end}}$ 为端点边界条件，$\mathcal{L}_{\text{bar}}$ 为均值约束，$\mathcal{L}_{\text{flatness}}$ 控制末端频谱平坦度[^src-stats]。

使用投影梯度下降 (PGD) 优化，定理 3.1 证明单调收敛到投影一阶驻点条件[^src-stats]。定理 3.2 证明当调度更新时前向漂移是 Lipschitz 稳定的，保证交替优化不破坏分布一致性[^src-stats]。

### FGD (Frequency Guided Denoiser)

FGD 估计调度诱导的频谱失真比率，作为条件信号调制去噪强度[^src-stats]。关键设计：在实例归一化空间中运行，与 STS 共享同分布空间[^src-stats]。

核心组件：**Spectral Conditioned Denoising Module (SCDM)**，使用多频带设计（默认 B=2）捕获粗粒度频率依赖失真，通过裁剪（r_min=-10, r_max=10）稳定条件信号[^src-stats]。

### 两阶段训练

- **阶段一**：交替训练 FGD（固定调度）和 STS（固定 FGD），逐步对齐前向腐蚀轨迹与去噪器容量[^src-stats]
- **阶段二**：冻结 STS，使用学习到的 β(t) 训练 FGD 至收敛，消除前向过程漂移[^src-stats]

## 关键结果

在 8 个多变量基准数据集（ECL, ILI, ETTh1, ETTh2, ETTm1, ETTm2, Traffic, SolarEnergy）上对比 5 个基线（CSDI, D3VAE, TimeDiff, DiffusionTS, NsDiff）[^src-stats]：

- **CRPS**: 在所有数据集上取得最佳，相对最佳基线改进 10.67%–17.43%[^src-stats]
- **MAE**: 在所有数据集上最佳[^src-stats]
- **MSE**: 大多数基准最佳；SolarEnergy 上略逊于 NsDiff（零值多、动态平滑，对振幅偏差敏感）[^src-stats]

**效率**：训练内存仅 27.74 MB (Traffic)，比 CSDI (3512 MB) 低 100 倍以上；训练时间 10.61 ms/iter，推理 4.82 ms/iter[^src-stats]。

**扩散步数敏感性**（表 3）：STS 在 T=10 时相比线性调度将 CRPS 降低最多 55.4%；随 T 增加优势递减并饱和[^src-stats]。

**消融**：移除 EPO（端点目标）退化最严重；移除 SDE（频谱失真估计）持续恶化 CRPS 和 MSE；移除 STS（回退线性调度）降低性能；移除 IN（实例归一化）明显退化[^src-stats]。

## 贡献

1. 首次将噪声调度学习和频率引导去噪联合优化用于时序扩散预测[^src-stats]
2. PGD 收敛性证明和调度更新的 Lipschitz 稳定性保证[^src-stats]
3. 频谱轨迹控制分析：STS 学习到数据特定的非单调调度模式（初期急剧上升→中期平坦→末端再次上升），避免在中间阶段过度压缩关键信息[^src-stats]
4. 极低计算开销（内存 27 MB），显著优于扩散基线[^src-stats]

[^src-stats]: [[source-stats]]
