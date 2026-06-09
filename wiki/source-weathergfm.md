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
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: WeatherGFM

**WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning** (Zhao et al., ICLR 2025; arXiv:2411.05420) introduces the first *weather generalist foundation model* that handles a diverse set of weather understanding tasks within a single unified model via visual in-context learning[^src-weathergfm].

## 核心问题

现有数据驱动天气模型聚焦单一任务（如天气预报），无法在统一模型中处理多样化复杂任务；且单场景、单观测设备依赖有限真实观测，限制了模型性能上限[^src-weathergfm]。此前的天气基础模型（如 ClimaX、Aurora）也主要面向预报和降尺度，未建模多模态多任务，且需要按任务微调[^src-weathergfm]。

## 方法

WeatherGFM 分三步统一天气理解任务[^src-weathergfm]：

1. **统一表示**：将各类任务统一抽象为从源数据到目标数据的投影 $\tau: X_S \to X_T$。当 $X_S=x_{LR}, X_T=x_{HR}$ 即空间超分；当 $X_S=\{x_1,\dots,x_t\}, X_T=\{x_{t+1},\dots\}$ 即天气预报[^src-weathergfm]。
2. **[[weather-prompt|天气提示设计]]**：针对单模态、多模态（跨通道）、时序三类输入分别设计三种视觉提示格式，每种提示给出示例 (input, target) 对加 query 输入[^src-weathergfm]。
3. **视觉提问-回答范式 + [[mixed-modal-masked-image-modeling|混合模态掩码图像建模 (MMIM)]]**：基于纯 ViT 主干，训练时对 prompt target 和 ground-truth target 按 75% 块状掩码比例做掩码，保留 prompt input 与 query input，用 MSE/L1 损失重建被掩码的 target；推理时完全掩码目标图像即可生成结果[^src-weathergfm]。任务特定 patch embedding 层 + MLP 对齐解决了不同任务通道数可变的问题[^src-weathergfm]。

## 贡献

- 提出首个天气通用基础模型，可处理 10+ 种天气理解任务（天气预报、超分/降尺度、天气图像翻译、后处理）[^src-weathergfm]。
- 天气提示设计支持时序、多模态、单模态数据[^src-weathergfm]。
- 首次展示天气基础模型对未见任务（OOD）的 in-context 泛化能力[^src-weathergfm]。

## 结果

在 SEVIR 与 POMINO-TROPOMI/GEOS-CF 数据上，单一 WeatherGFM（110M base / 330M large）在 10 个任务上多数优于各任务单独训练的 UNet/ViT 单任务模型，验证了统一建模可突破单任务性能上限[^src-weathergfm]。展示了数据与模型规模的 scaling law；OOD 实验（IR107 外推、IR107→IR069 翻译、15 分钟时序超分）显示对类训练分布任务能直接泛化，但对差异大的多卫星空间超分失败[^src-weathergfm]。ERA5 可扩展性实验中单一模型预测 7 个 lead time 的 T2m，在 120h/168h 上优于 ClimaX 甚至 ECMWF IFS，且收敛更快[^src-weathergfm]。

## 局限

- 评估指标主要为 RMSE 与 CSI；作者明确表示目标不是在每个任务上达到 SOTA[^src-weathergfm]。
- 集合/概率预报与不确定性量化未涉及（判别式 MSE/L1 训练）[^src-weathergfm]。
- OOD 泛化对与训练分布差异大的任务（多模态空间超分）失效[^src-weathergfm]。

[^src-weathergfm]: [[source-weathergfm]]
