---
title: "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-imputation
  - self-supervised-learning
  - neurips-2021
created: 2026-05-31
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Source: CSDI

**作者**: Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon (Stanford University; Mitsubishi UFJ Trust Investment Technology Institute; Japan Digital Design)
**发表**: NeurIPS 2021
**领域**: 多元时间序列概率插补

## 核心论点

CSDI 是首个将条件扩散模型显式用于时间序列缺失值插补的工作[^src-csdi]。此前利用扩散模型做插补的方法（Song et al., 2021; Kadkhodaie & Simoncelli, 2021）采用"事后修补"策略——用预训练的无条件扩散模型，在采样时将已知观测值硬注入生成结果。CSDI 的洞察是：既然目标是对条件分布 $p(x^{\text{ta}} \mid x^{\text{co}})$ 采样，就应该在训练阶段直接让去噪网络学习这个条件分布，而非事后近似。为此 CSDI 做了三件事：(1) 将 [[ddpm|DDPM]] 的去噪函数 $\epsilon_\theta(x_t, t)$ 扩展为条件形式 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$；(2) 设计自监督训练策略，从观测值中人工构造"伪缺失目标"和"伪条件观测"；(3) 用时间 Transformer + 特征 Transformer 的双轴注意力替代 DiffWave 的膨胀卷积，分别捕获时间依赖和跨特征依赖[^src-csdi]。

## 方法

- **条件扩散**：将 DDPM 的参数化 $\mu_\theta(x_t, t)$ 直接扩展到条件场景 $\mu_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$，训练目标仍然是 MSE 噪声预测，形式完全不变——唯一的区别是 $\epsilon_\theta$ 多了一个条件输入 $x_0^{\text{co}}$
- **自监督训练**：受 BERT 掩码语言建模启发，从训练样本的观测值中随机选一部分作为"伪插补目标" $x_0^{\text{ta}}$，其余作为"伪条件观测" $x_0^{\text{co}}$，在伪目标上加噪声后训练去噪网络。提供四种目标选择策略（Random / Historical / Mix / Test pattern）以适配不同缺失模式
- **双轴注意力架构**：每层残差块包含两个 1 层 Transformer 编码器——时间 Transformer 沿特征轴学习时间依赖，特征 Transformer 沿时间轴学习跨特征依赖。4 层残差层，残差通道 C=64，8 注意力头，约 415K 参数。T=50，二次方噪声调度

## 关键结果

- **概率插补**：在 PhysioNet 医疗数据（35 特征，48 时间步，~80% 缺失率）和北京空气质量数据（36 特征，36 时间步，~13% 缺失率）上，CSDI 的 CRPS 比 GP-VAE 降低 40-65%[^src-csdi]
- **确定性插补**：MAE 比 BRITS/GLIMA 降低 5-20%
- **不规则采样插值**：CRPS 大幅领先 Latent ODE 和 mTANs
- **预测**：在 electricity 和 traffic 数据集上超越 [[timegrad|TimeGrad]]，在所有数据集上具有竞争力

## 贡献

1. 开创了"条件扩散模型 + 自监督训练 + 时间序列插补"的方向，证明条件建模相比无条件扩散+事后约束有实质性收益（实验差值约 28% CRPS 改善）
2. 证明 T=50 步扩散对时间序列插补足够，远低于图像的 T=1000
3. 为后续工作（SSSD、[[tsdiff|TSDiff]] 中的条件对照、CSDI 变体等）奠定了标准范式；TSDiff 在预测基准上直接与 CSDI 比较，并强调无条件 + 推理引导的任务无关替代路线

## 局限性

- **计算效率**：T=50 步仍需串行推理，实时场景受限
- **注意力 $O(L^2)$ 复杂度**：长序列时计算成本膨胀（后续 SSSD 用 S4 状态空间模型替代 Transformer 作为回应）
- **假设观测值可靠**：未处理传感器故障导致的错误观测值被当作条件注入的风险
- **仅验证时间序列**：声称框架不限于时序，但未在表格数据或图像修复上验证

[^src-csdi]: [[source-csdi]]
