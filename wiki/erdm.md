---
title: "ERDM: Elucidated Rolling Diffusion Models"
type: technique
tags:
  - diffusion-models
  - probabilistic-forecasting
  - spatiotemporal-forecasting
  - weather-forecasting
  - fluid-dynamics
  - neurips-2025
  - edm
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# ERDM

**ERDM**（Elucidated Rolling Diffusion Models）是 Cachay 等（UC San Diego & NVIDIA）在 NeurIPS 2025 提出的扩散概率预测框架，首次将 Rolling Sequence Diffusion (RSDM) 的滚动预测结构与 EDM 的高保真设计空间统一[^src-erdm]。

## 核心设计

### 从 RSDM 到 EDM

传统 RSDM 基于 DDPM，未利用 EDM 的 principled design choices（预处理、二阶采样、损失权重）。ERDM 将三个核心 EDM 组件适配到滚动预测设置：

1. **噪声 schedule 适配**：窗口内 snapshot w 的噪声水平 σ̄w(t) 从 σmin 单调递增至 σmax，曲率参数 ρ=−10（vs EDM ρ=7），确保近未来低噪声、远未来高噪声[^src-erdm]。训练时固定 schedule，随机化导致 ∼2× 性能下降。

2. **网络预处理向量化**：cskip、cout、cin、cnoise 按 snapshot 独立应用，适配 snapshot-dependent noise levels[^src-erdm]。

3. **Heun 二阶采样**：概率流 ODE 驱动 W 个耦合扩散过程同时演化，每完成一次 t:0→1 积分即输出首个 denoised snapshot，剩余窗口前移并追加新纯噪声 snapshot[^src-erdm]。

### 不确定性感知损失重加权

EDM 的 λ(σ) 确保网络目标为单位方差。ERDM 额外乘以对数正态 PDF f(σ; Pmean, Pstd)，聚焦中等噪声水平——从确定性到随机性的过渡区间——使模型容量集中在最关键的中程预测时域[^src-erdm]。

### 混合 3D 时空架构

在 2D ADM U-Net 的每个上下采样块前后嵌入 causal temporal attention 层，噪声嵌入同时注入空间和时间路径。相比简单将时间维堆叠为通道（4× 性能退化），该架构显式保留时间结构[^src-erdm]。

### 首个窗口初始化

用预训练 EDM 预测首个窗口 ŷ1:W，叠加 schedule 匹配的噪声 x̄w ~ N(ŷw, σ̄w(0)²) 后开始滚动采样，避免 RSDM 原需的双任务联合训练（去噪 + 自初始化）[^src-erdm]。

## 实验结果

| 指标 | Navier-Stokes (64步) | ERA5 1.5° |
|------|---------------------|-----------|
| CRPS vs EDM | 50% 提升 | 达 10% 提升 |
| 校准 (SSR) | 一致优于 EDM | 与 IFS ENS 持平 |
| 物理真实度 | — | 功率谱与 IFS ENS 持平 |

Navier-Stokes 上 ERDM 显著超越 DYffusion 和 PDE-Refiner。ERA5 上中长程与 IFS ENS 和 NeuralGCM ENS 竞争，但训练仅需 4 H200 GPU × 5 天（vs NeuralGCM 128 TPUv5 × 10 天）[^src-erdm]。

## 关键消融

- 移除损失重加权：>2× CRPS 退化
- 2D 通道堆叠替代 3D hybrid 架构：4× 退化
- ρ 偏离 [−30, −5]：显著退化
- 窗口大小 W 和噪声边界 σmin/σmax 在合理范围内鲁棒[^src-erdm]

## 与其他方法的关系

- **vs [[edm|EDM]]**：ERDM 是 EDM 在序列滚动预测场景的扩展，保留 EDM 预处理/采样/损失框架并适配渐进噪声[^src-erdm]
- **vs [[dyffusion|DYffusion]]**：DYffusion 用插值替代高斯噪声，ERDM 保留高斯噪声框架但引入 snapshot-dependent 渐进噪声和 EDM design[^src-erdm]
- **vs [[tedm|TEDM]]**：两者都基于 EDM 框架，TEDM 将扩散时间轴 = 物理时间轴（O(H) 采样），ERDM 用滚动窗口 + 渐进噪声显式建模不确定性传播[^src-erdm]
- **vs GenCast**：ERDM 的 EDM baseline 可视为 GenCast 复现；ERDM 在此基础上引入滚动扩散机制[^src-erdm]

## 局限

3D denoiser 内存开销大（推理 49GB vs EDM 21GB），短程预测弱于 IFS ENS，依赖外部模型初始化，损失权重可能逊于 importance sampling[^src-erdm]。

## 相关页面

- [[source-erdm]] — 论文摘要
- [[rolling-diffusion-models]] — 滚动扩散模型概念
- [[edm]] — EDM 框架
- [[dyffusion]] — DYffusion 对比
- [[tedm]] — TEDM 对比
- [[diffusion-model]] — 扩散模型总论

[^src-erdm]: [[source-erdm]]
