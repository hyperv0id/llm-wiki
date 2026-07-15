---
title: "ERDM: Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics"
type: source-summary
tags:
  - diffusion-models
  - probabilistic-forecasting
  - spatiotemporal-forecasting
  - weather-forecasting
  - fluid-dynamics
  - neurips-2025
created: 2026-07-16
last_updated: 2026-07-16
source_count: 0
confidence: medium
status: active
---

# ERDM: Elucidated Rolling Diffusion Models

**Salva Rühling Cachay (UC San Diego), Arash Vahdat, Miika Aittala, Karsten Kreis, Morteza Mardani, Noah Brenowitz (NVIDIA), Rose Yu (UC San Diego), NeurIPS 2025**

## 核心贡献

ERDM 首次将 Rolling Sequence Diffusion (RSDM) 的滚动预测结构与 EDM 的高保真扩散设计空间统一，提出三个关键贡献：

1. **不确定性感知的损失重加权**：在对数正态分布 f(σ) 下聚焦中等噪声水平（deterministic→stochastic 的过渡区间），使模型容量集中在最关键的预测时域。
2. **高效初始化策略**：用预训练 EDM 提供首个窗口的预测，叠加与滚动 schedule 匹配的噪声，避免 RSDM 原有的双任务联合训练开销。
3. **混合 3D 时空架构**：2D ADM U-Net 嵌入 causal temporal attention 层，噪声嵌入同时注入空间和时间路径。

## 技术要点

- 将 EDM 的噪声 schedule 适配为窗口内 snapshot-dependent 的渐进噪声 σ̄w(t)，曲率 ρ=−10（vs EDM 默认 ρ=7）确保近未来低噪声、远未来高噪声。
- 训练时固定 schedule（非随机化），随机化训练导致 ∼2× 性能下降。
- 概率流 ODE 以 Heun 二阶求解器采样，W=6 窗口，N=1.25~2 步/snapshot。
- 使用时间相关噪声先验（Ge et al., 2024, α=1）替代 i.i.d. 噪声，略微改善长程预测。

## 实验

- **Navier-Stokes**：ERDM 在 64 步 rollout 上实现 50% CRPS 提升（vs EDM W=4），校准度显著优于所有 EDM baseline。
- **ERA5 1.5° 天气预测**：CRPS 相对 EDM 提升达 10%，中长程与 IFS ENS/NeuralGCM ENS 竞争，功率谱物理真实度与 IFS ENS 持平。仅需 4 H200 GPU × 5 天训练（vs NeuralGCM 128 TPUv5 × 10 天）。
- 消融：架构从 2D 通道堆叠改为 3D hybrid 提升 4×；损失重加权移除导致 >2× 退化；ρ 偏离 [-30,-5] 范围显著降低性能。

## 局限

3D 架构内存开销大（推理 49GB vs EDM 21GB），短程预测弱于 IFS ENS，依赖外部模型初始化首个窗口，噪声水平损失加权可能逊于 importance sampling。
