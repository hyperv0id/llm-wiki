---
title: "Swift"
type: entity
tags:
  - weather-forecasting
  - consistency-models
  - probabilistic-forecasting
  - ensemble
  - era5
  - arxiv-2025
  - diffusion-models
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: medium
status: active
---

# Swift

**Swift** 是首个**自回归一致性模型**用于概率天气预测，由 Argonne National Laboratory 的 Stock、Arcomano 和 Kotamarthi 提出（arXiv:2509.25631, 2025 年 9 月）[^src-swift]。它通过单步采样实现 39× 推理加速，同时维持 75 天稳定预报，弥合中程预测到季节尺度的差距。

## 核心设计

### 单步一致性采样

Swift 基于 TrigFlow 框架（统一 EDM 和 Flow Matching），将一致性函数参数化为 $f_\theta(x_t, t) = \cos(t)x_t - \sin(t)\sigma_d F_\theta(x_t/\sigma_d, t)$。在推理时，$t = \pi/2$ 的单步前向传播即可从噪声生成样本，替代扩散模型所需的 20–40 NFE[^src-swift]。

### CRPS 自回归微调

Swift 的 **[[crps-autoregressive-finetuning|CRPS 自回归微调]]** 是使其区别于图像一致性模型的关键创新。预训练后，模型在多步自回归 rollout（K=1–8）上通过 CRPS 损失微调，直接优化集合校准度。CRPS 平衡预测精度（第一项）和集合散布（第二项），生成物理上校准的集合预报[^src-swift]。

### 架构

- **225M 参数** Swin Transformer，非层级结构
- 每两层交替 x- 和 y-方向移位窗口
- adaLN 调制层、SwiGLU FFN
- 2×2 patch 嵌入，隐藏维度 1056，12 注意力头，12 注意力块
- 动态时间间隔 δi ∼ U{6, 12, 24} 正则化训练动态[^src-swift]

## 数据与任务

- **ERA5** 0–4 年代再分析数据（WeatherBench 2），1.40625° 分辨率（128×256 像素）
- 4 地面变量 + 5 大气变量 × 13 气压层（50–1000 hPa）
- 学习 $p(x_{i+1} \mid x_i)$，残差预测 $x_{\delta_i} - x_i$[^src-swift]

## 关键结果

| 指标 | Swift | Diffusion baseline | GenCast |
|------|-------|-------------------|---------|
| NFE/步 | **1** | 39 | 20–40 |
| 中程 RMSE | 竞争 IFS ENS | 相似 | 最优 |
| 稳定预报 | **75 天** | 15 天 | ~15 天 |
| 推理速度 | **15s/64 预报** | 7.6min/12 预报 | — |

- Hurricane Laura（2020-08-27）48 成员集合捕获了真实风暴轨迹和大气河流[^src-swift]
- 75 天 Hovmöller 图重现赤道波模传播[^src-swift]
- 外热带季节循环：南北半球温度趋势与 ERA5 一致[^src-swift]

## 局限

- 预报散布不足（SSR < 1），起步 0–4 天尤为明显
- 极地区域误差较大，受纬度加权损失和 ERA5 再分析不确定性影响
- 平滑变量（z500, mslp）在高纬向波数漂移到大尺度，可通过场截断缓解[^src-swift]

## 未来方向

- 从更大、高分辨率扩散模型（如 Aeris）蒸馏一致性模型
- Classifier-free guidance 改善预报性能
- 贪婪调度（Pangu-Weather 风格）改善高时间分辨率稳定性[^src-swift]

## 相关页面

- [[consistency-models]] — 一致性模型基础
- [[autoregressive-consistency-models]] — 自回归一致性模型概念
- [[crps-autoregressive-finetuning]] — CRPS 微调技术
- [[trigflow]] — TrigFlow 统一框架
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[diffusion-models]] — 扩散模型
- [[probability-flow-ode]] — 概率流 ODE
- [[ensemble-forecasting-calibration]] — 集合预报校准
- [[uniextreme]] — 通用极端天气基础模型
- [[extreme-weather-forecasting]] — 极端天气预测
- [[weathergfm|WeatherGFM]] — 天气通用基础模型，用视觉 in-context learning 统一 10+ 种天气理解任务；与 Swift 互补——Swift 专注生成式概率集合预报，WeatherGFM 为判别式多任务统一框架[^src-weathergfm]。

[^src-swift]: [[source-swift]]
[^src-weathergfm]: [[source-weathergfm]]
