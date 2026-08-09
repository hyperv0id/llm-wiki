---
title: "Source: WeatherGFM — Learning A Weather Generalist Foundation Model via In-context Learning"
type: source-summary
tags:
  - weather-forecasting
  - foundation-model
  - in-context-learning
  - vision-transformer
  - multi-task
  - iclr-2025
created: 2026-06-08
last_updated: 2026-08-09
source_count: 0
confidence: low
status: active
---

# Source: WeatherGFM

**WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning**（Zhao 等，ICLR 2025；arXiv:2411.05420）。论文提出首个天气通用基础模型，用视觉 in-context learning 在单一 ViT 内统一 10+ 种天气理解任务（预报、超分/降尺度、图像翻译、后处理）。

## 核心问题

论文指出，数据驱动天气模型多为单任务、依赖单场景有限观测，性能受限；ClimaX、Aurora 等天气基础模型主要面向预报/降尺度且需按任务微调。核心问题是"能否用单一通用模型处理多样天气任务与数据模态"。

## 方法

1. **统一表示**：任务抽象为源到目标的投影 $\tau: X_S \to X_T$（空间超分 $x_{LR}\to x_{HR}$；天气预报 $\{x_t\}\to\{x_{t+1},\dots\}$）。
2. **[[weather-prompt|天气提示]]**：针对单模态、多模态（跨通道）、时序三类输入设计三种视觉提示格式，示例 (input, target) 对 + query 输入决定任务语义。
3. **[[mixed-modal-masked-image-modeling|MMIM]]**：纯 ViT 主干，训练时按 75% 块状掩码重建 prompt target 与 ground-truth target（保留 prompt/query input），推理时目标全掩码生成；任务特定 patch embedding + MLP 对齐适配可变通道数。

## 贡献（论文自述）

- 论文提出首个天气通用基础模型，覆盖 10+ 种天气理解任务。
- 论文首次展示天气基础模型对未见任务（OOD）的 in-context 泛化。

## 结果（论文报告）

在 SEVIR 与 POMINO-TROPOMI/GEOS-CF 设置下，作者报告单一模型（110M/330M）在 10 个任务上多数优于各任务单训的 UNet/ViT，并报告数据与模型规模的 scaling law；OOD 上对类训练分布任务可直接泛化，对差异大的多卫星空间超分失效。ERA5 上单模型预测 7 个 lead time 的 T2m，120h/168h 优于 ClimaX 与 ECMWF IFS 且收敛更快。

## 局限（论文自述）

- 判别式 MSE/L1 训练，未涉及集合/概率预报与不确定性量化。
- 作者明确不追求每任务 SOTA，重点在通用性与 in-context 泛化。
- OOD 泛化对与训练分布差异大的任务失效。

## 相关页面

- [[weathergfm]] — WeatherGFM 主模型
- [[weather-prompt]] — 天气提示设计
- [[mixed-modal-masked-image-modeling]] — MMIM 训练-推理范式
- [[weather-foundation-model]] — 天气基础模型概念
