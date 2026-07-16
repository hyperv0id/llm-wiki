---
title: "Stochastic Fisher-Guided Adaptive Selection"
type: technique
tags:
  - weather-forecasting
  - parameter-efficient-fine-tuning
  - fisher-information
  - parameter-selection
  - stochastic-regularization
  - iclr-2026
created: 2026-07-25
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# Stochastic Fisher-Guided Adaptive Selection (SFAS)

**Stochastic Fisher-Guided Adaptive Selection（SFAS）** 是 [[weatherpeft|WeatherPEFT]] 框架的反向传播组件，利用 Fisher 信息矩阵量化参数重要性并结合退火随机干预，识别并仅更新对下游任务最关键的 Top-k 参数[^src-weatherpeft]。

## 设计动机

不同天气下游任务对预训练骨干的需求不同：降水预报可能需要混沌模式相关参数，而降尺度任务依赖空间关系参数[^src-weatherpeft]。通用 PEFT 方法（如 LoRA）对所有任务应用相同的低秩更新，task-selective 方法（如 SCT、Child-Tuning）采用静态预训练选择，无法在训练中动态重校准。SFAS 通过 Fisher 信息提供原则性的参数重要性度量，并引入退火随机性稳定选择[^src-weatherpeft]。

## 方法

### Fisher 信息近似

参数 θ 的重要性以对其微扰 δ 引起输出分布变化的 KL 散度衡量[^src-weatherpeft]：

$$\mathbb{E}_X[D_{KL}(P_\theta(Y|X) \parallel P_{\theta+\delta}(Y|X))] = \delta^T F_\theta \delta + O(\delta^3)$$

其中 $F_\theta$ 为 Fisher 信息矩阵。实际使用对角近似[^src-weatherpeft]：

$$\hat{F}_\theta = \frac{1}{N}\sum_{j=1}^N (\nabla_\theta \log P_\theta(Y_j|X_j))^2$$

每个参数的 $\hat{F}_\theta$ 值越大，表示其对学习目标越敏感，越值得更新[^src-weatherpeft]。

### 退火随机干预

天气下游任务的异质性导致训练早期存在大量噪声，高 Fisher 值的参数可能仅捕获瞬态噪声而非任务相关特征[^src-weatherpeft]。SFAS 引入退火随机分量[^src-weatherpeft]：

$$\bar{F}_\theta = \gamma \times (1 - \frac{n_s}{t_s}) \odot M_{sc} + \hat{F}_\theta$$

其中 $M_{sc} \sim \text{Uniform}(0,1)$ 为随机向量，$\gamma$ 为初始因子，$n_s/t_s$ 随训练步数从 1 线性衰减至 0。训练初期随机性高，逐步退火至纯 Fisher 引导[^src-weatherpeft]。

### Top-k 选择

每批次选择 $\bar{F}_\theta$ 最大的 k 个参数生成 Fish Mask（选中位置为 1，其余为 0），仅更新被选中的参数[^src-weatherpeft]。超参数 k 控制稀疏度。

## 消融结果

单独使用 SFAS 在降尺度任务上同样超越标准 PEFT 基线，验证了任务自适应参数选择的价值[^src-weatherpeft]。在区域降水预报中，SFAS 比 TADP 贡献更大——降水的稀疏、局部化信号天然更依赖参数级选择而非特征级提示[^src-weatherpeft]。完整 WeatherPEFT（TADP+SFAS）在三任务上均取得最优，验证了前向-反向双阶段协同[^src-weatherpeft]。

## 与相关技术的关系

- **Child-Tuning / SAM / SCT**：均为 task-selective PEFT，但使用静态预训练选择策略，缺乏 SFAS 的 Fisher 引导 + 退火随机性的动态重校准机制[^src-weatherpeft]
- **EWC（Elastic Weight Consolidation）**：同样使用 Fisher 信息，但用于持续学习中防止灾难性遗忘；SFAS 则用于选择应更新的参数
- **[[projected-fisher-divergence|投影 Fisher 散度]]**：Flux Matching 中的统计散度概念，与 SFAS 的 Fisher 信息选择目的不同

## 相关页面

- [[weatherpeft]] — WeatherPEFT 完整框架
- [[task-adaptive-dynamic-prompting]] — TADP 前向传播组件
- [[source-weatherpeft]] — 源文件摘要
- [[weather-foundation-model]] — 天气基础模型概念

[^src-weatherpeft]: [[source-weatherpeft]]
