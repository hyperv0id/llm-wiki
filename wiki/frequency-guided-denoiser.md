---
title: "Frequency Guided Denoiser (FGD)"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - denoising
  - time-series
  - spectral-analysis
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Frequency Guided Denoiser (FGD)

**FGD** (Frequency Guided Denoiser) 是 [[stats|StaTS]] 框架中的频率引导去噪组件，通过估计调度诱导的频谱失真来调制去噪强度[^src-stats]。

## 核心思想

不同扩散步和不同变量的调度诱导频谱破坏程度不同。FGD 估计这种失真，将其作为条件信号注入去噪网络，实现异构恢复——对受损严重的频率分量施加更强的修正[^src-stats]。

关键假设：去噪器如果能"感知"当前状态相比原始数据丢失了哪些频谱信息，就能更有针对性地恢复[^src-stats]。

## SCDM 架构

**Spectral Conditioned Denoising Module (SCDM)** 是 FGD 的核心[^src-stats]：

### 频谱失真估计

1. 计算当前加噪状态 $x_t$ 的频谱
2. 计算调度诱导的频谱衰减比率（相对于干净数据频谱）
3. 归一化并裁剪到 [-10, 10] 稳定条件信号[^src-stats]

### 多频带设计（B=2 默认）

将频谱分成 B 个频带，每个频带独立估计失真比率，捕获粗粒度频率依赖的失真模式[^src-stats]。这种设计使模型能区分低频和高频成分的不同腐蚀速率。

### 条件注入

估计的频谱失真信息通过条件归一化层注入去噪网络的各层，动态调制每层、每步、每变量的去噪强度[^src-stats]。

## 实例归一化空间

FGD 在实例归一化（RevIN）空间中运行，与 STS 共享同分布空间[^src-stats]。实例归一化利用历史窗口的统计量对齐特征分布，缓解振幅偏移[^src-stats]。消融显示移除 IN 导致所有基准上性能明显退化，尤其在 ETTm1 上退化最严重——表明特征分布对齐对去噪稳定性至关重要[^src-stats]。

## 与 STS 的协同

FGD 和 STS 形成正反馈循环[^src-stats]：

1. **STS 学习调度** → 产生特定频谱衰减轨迹
2. **FGD 感知衰减** → 针对性调整去噪强度
3. **联合优化** → 调度与去噪器容量逐步对齐

两阶段交替训练确保这个循环不会发散[^src-stats]：
- 阶段一：交替固定一方训练另一方
- 阶段二：固定 STS → FGD 训练至收敛

## 频谱轨迹可控性

FGD 与 STS 协同改进了频谱衰减和重建的逐步对齐[^src-stats]：

- **T=10 时**：前向过程将频谱压缩集中到少数跃迁步，反向链重构机会少——FGD 的调度感知变得关键
- **T=50 时**：频谱衰减和恢复分布到更多步，产生更平滑连续的重建——FGD 的频带条件信号帮助维持逐步一致性

在任何 T 下，STS+FGD 组合都比固定线性调度产生更可控的频谱演化和更稳定的条件生成[^src-stats]。

## 消融

移除 SDE（频谱失真估计）持续恶化 CRPS 并增加 MSE——表明去噪器从调度诱导的频谱损伤信息中受益，能以异构腐蚀下调制修正[^src-stats]。

[^src-stats]: [[source-stats]]
