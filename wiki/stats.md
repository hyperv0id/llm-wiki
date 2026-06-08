---
title: "StaTS"
type: entity
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

# StaTS

**StaTS** (Spectral Trajectory Schedule Learning for Adaptive Time Series Forecasting with Frequency Guided Denoiser) 是一个联合优化噪声调度和去噪过程的扩散概率时间序列预测框架，发表于 arXiv 2603[^src-stats]。

## 核心思想

StaTS 由两个协同组件构成：

1. **[[spectral-trajectory-scheduler|Spectral Trajectory Scheduler (STS)]]**：通过频域正则化学习数据自适应的噪声调度 β(t)，产生与数据集频谱特征对齐的前向腐蚀轨迹[^src-stats]
2. **[[frequency-guided-denoiser|Frequency Guided Denoiser (FGD)]]**：估计调度诱导的频谱失真，作为条件信号调制去噪强度跨扩散步和变量[^src-stats]

关键发现：学习到的 β(t) 偏离标准单调模板，呈现明显的非线性——初期急剧上升、中期平坦、末端再次上升。这避免了在中间阶段过度压缩关键频谱信息[^src-stats]。

## 架构

```
输入 X (历史序列)
    │
    ├──► STS (频谱轨迹调度器)
    │       学习自适应噪声调度 β(t)
    │       频域正则化 (平滑/端点/均值/平坦度)
    │       └──► 产生腐蚀轨迹 ᾱt
    │
    ├──► FGD (频率引导去噪器)
    │       ├── SCDM (频谱条件去噪模块)
    │       │     多频带频谱失真估计
    │       │     失真比率裁剪 (r∈[-10,10])
    │       └── 在实例归一化空间中
    │
    ▼
逆扩散过程: Y_T → Y_{T-1} → ... → Y_0
    │  T=50 步默认
    ▼
    Y_0 — 预测的多变量序列
```

## 训练策略

两阶段交替优化[^src-stats]：

- **阶段一**：交替优化。固定 FGD → 更新 STS（PGD 优化，定理 3.1 保证收敛）。固定 STS → 训练 FGD。逐步对齐前向轨迹与去噪能力[^src-stats]。
- **阶段二**：冻结 STS → 训练 FGD 至收敛（稳定优化，消除前向过程漂移）[^src-stats]。

STS 和 FGD 在同一实例归一化空间中运行以保证分布一致性[^src-stats]。定理 3.2 保证调度更新时前向漂移是 Lipschitz 稳定的[^src-stats]。

## 性能

在 8 个多变量数据集上对比 5 个概率预测基线[^src-stats]：

| 指标 | StaTS vs 最佳基线 |
|------|-------------------|
| CRPS | 改进 10.67%–17.43%（全部最佳） |
| MAE  | 全部最佳 |
| MSE  | 7/8 最佳（SolarEnergy 例外） |
| 训练内存 | 27.74 MB（vs CSDI 3512 MB） |

**扩散步效率**：T=10 时 CRPS 相比线性调度降低 55.4% (ETTm1)，T=50 时优势缩小至 3%[^src-stats]。STS 在小扩散步预算时收益最大[^src-stats]。

## 与基线方法关系

- **[[nsdiff|NsDiff]]**：StaTS 当前最强扩散基线对手。NsDiff 使用非平稳 LSNM+UANS 建模时变不确定性，StaTS 通过频谱轨迹调度+频率引导去噪在所有 CRPS 和 MAE 上超越[^src-stats]。MSE 分布在 ETTm1 和 Electricity 上 StaTS 更集中且左偏，表示更多测试实例的预测误差更低[^src-stats]
- **[[timegrad|TimeGrad]]**：首个时序扩散模型，使用自回归 RNN+LSTM 编码 + 固定线性调度。StaTS 的关键进展是学习数据自适应调度替代固定调度[^src-stats]
- **CSDI**：非自回归条件得分扩散。StaTS 以更低内存（27 MB vs 3512 MB）实现更好性能[^src-stats]
- **TimeDiff**：使用未来 mixup 和自回归初始化的时序扩散。StaTS 在所有指标上超越[^src-stats]
- **DiffusionTS**：可解释扩散时序生成。StaTS 显著超越[^src-stats]
- **[[d3vae|D3VAE]]**：扩散-去噪-解耦耦合框架。StaTS 全面超越[^src-stats]

## 局限性

1. arXiv 预印本，未经同行评审[^src-stats]
2. SolarEnergy 数据集上 MSE 略逊于 NsDiff（零值多、动态平滑）[^src-stats]
3. 两阶段训练增加了训练流程复杂性[^src-stats]

[^src-stats]: [[source-stats]]
