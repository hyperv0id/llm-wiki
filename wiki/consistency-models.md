---
title: "Consistency Models"
type: entity
tags:
  - diffusion-models
  - fast-inference
  - one-step-generation
  - icml-2023
created: 2026-04-28
last_updated: 2026-08-29
source_count: 5
confidence: medium
status: active
---

# Consistency Models

**Consistency Models** 是扩散模型加速采样的里程碑工作，由 OpenAI 的 Yang Song 等人于 2023 年发表在 ICML。该模型支持单步生成，同时保留多步采样和零样本编辑能力[^src-consistency-models]。

## 核心创新

1. **单步生成**：直接映射噪声到数据，无需迭代
2. **自一致性**：同一 PF ODE 轨迹上的点映射到相同起点
3. **双训练模式**：蒸馏模式 (CD) + 独立训练模式 (CT)
4. **零样本编辑**：无需显式训练即可执行多种图像编辑任务

## 技术细节

### 定义

给定 PF ODE 的解轨迹 $\{x_t\}_{t \in [\epsilon, T]}$，一致性函数定义为：
$$
f: (x_t, t) \mapsto x_\epsilon
$$

### 参数化

使用 skip connection 参数化：
$$
f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)
$$

其中 $c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$（边界条件）

### 采样

**单步采样**：
$$
x_0 = f_\theta(x_T, T), \quad x_T \sim \mathcal{N}(0, T^2 I)
$$

**多步采样**：通过交替去噪和噪声注入提升质量

## 实验结果

- **CIFAR-10**: 1 步 FID 3.55, 2 步 FID 2.93
- **ImageNet 64×64**: 1 步 FID 6.20, 2 步 FID 4.70

## MeanFlow 原文对 CM 系的定位

[[meanflow|MeanFlow]]（arXiv 2025）作者将 Consistency Models 系刻画为：一致性约束施加在网络行为上而非底层 ground-truth 场的性质上、路径锚定数据侧（在 MeanFlow 记法下相当于固定 $r\equiv 0$），网络只条件化单一时间变量；并认为此类训练可能不稳定、需要"仔细设计的离散化课程"逐步约束时间域——这是 MeanFlow 作者的表述[^src-meanflow]。MeanFlow 自身则由平均速度定义导出恒等式，条件化 $(r,t)$ 两个时间变量，不依赖额外一致性启发式[^src-meanflow]。

后续 [[improved-meanflows|iMF]]（MeanFlow 团队，arXiv:2512.02012 v2）将 CM 系概括为 fastforward generative models 中"从中间时刻直接跳到轨迹终点"的一类（iMF Sec. 2），并在其 1-NFE 从头训练对照中报告 iCT-XL/2 FID 34.24、iMF-XL/2 1.72（iMF 论文 Tab. 3，作者报告）[^src-improved-meanflows]。

## 应用扩展

### 自回归一致性模型

[[swift|Swift]] (arXiv 2025) 是将一致性模型从独立样本生成扩展到自回归序列预测的开创性工作[^src-swift]。Swift 首次将一致性模型应用于天气预测——每一步的单步采样（NFE=1）替代扩散模型的 20–40 NFE，使多步自回归微调成为可能。通过 [[crps-autoregressive-finetuning|CRPS 自回归微调]]，Swift 在 75 天稳定预报中实现 39× 加速，与 IFS ENS 集合系统竞争[^src-swift]。详见 [[autoregressive-consistency-models|自回归一致性模型]]。

[[loft|LOFT]]（KDD 2026）将速度一致性思想引入时空插补：在 CFM 上加速度一致性目标并用 [[uncertainty-aware-rectification|不确定性感知矫正]] 仲裁精度-线性化的梯度冲突，2 NFE 完成推理；论文报告静态施加该约束（[[consistency-fm|Consistency-FM]]/[[alphaflow|AlphaFlow]] 式）在高稀疏目标下因梯度冲突而退化[^src-loft]。详见 [[trajectory-consistency-flow-matching|轨迹一致性流匹配]]。

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[score-based-sde]] — Score-Based SDE，理论基础
- [[probability-flow-ode]] — 概率流 ODE
- [[dpm-solver]] — DPM-Solver，另一快速采样方法
- [[instaflow]] — InstaFlow，reflow+distill 路线的另一种一步生成方法 (ICLR 2024)
- [[rectified-flow|Rectified Flow]] — 直线 ODE 生成，与一致性模型同为少步生成方法
- [[swift]] — Swift，首个自回归一致性模型用于天气预测 (arXiv 2025)
- [[autoregressive-consistency-models]] — 自回归一致性模型概念
- [[crps-autoregressive-finetuning]] — CRPS 自回归微调技术
- [[loft]] — LOFT，速度一致性 + 不确定性矫正的时空插补 (KDD 2026)
- [[consistency-fm]] — Consistency-FM，速度一致性直线流 (arXiv 2024)
- [[alphaflow]] — α-Flow，统一 FM/Shortcut/MeanFlow 的目标族 (arXiv 2025)
- [[meanflow]] — MeanFlow，区间平均速度少步生成框架 (arXiv 2025)
- [[improved-meanflows]] — iMF，MeanFlow 后续改进，1-NFE 从头训练 FID 1.72（作者报告）(arXiv 2025)
- [[trajectory-consistency-flow-matching]] — 轨迹一致性流匹配技术页

## 引用

[^src-consistency-models]: [[source-consistency-models]]
[^src-swift]: [[source-swift]]
[^src-loft]: [[source-loft]]
[^src-meanflow]: [[source-meanflow]]
[^src-improved-meanflows]: [[source-improved-meanflows]]