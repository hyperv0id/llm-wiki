---
title: "WeatherGFM"
type: entity
tags:
  - weather-forecasting
  - foundation-model
  - in-context-learning
  - vision-transformer
  - multi-task
  - super-resolution
  - iclr-2025
created: 2026-06-08
last_updated: 2026-08-09
source_count: 1
confidence: medium
status: active
---

# WeatherGFM

**WeatherGFM** (Weather Generalist Foundation Model) 由 Zhao 等人（上海 AI Lab、香港理工大学等）提出，论文将其定位为首个天气通用基础模型（ICLR 2025, arXiv:2411.05420）[^src-weathergfm]。它通过**视觉 in-context learning** 在单一模型内统一处理 10+ 种天气理解任务，包括天气预报、超分辨率（降尺度）、天气图像翻译和后处理[^src-weathergfm]。

## 核心动机

论文指出，现有天气模型聚焦单一任务且依赖单场景有限观测，限制了性能上限；ClimaX、Aurora 等天气基础模型主要面向预报/降尺度，未建模多模态多任务且需按任务微调[^src-weathergfm]。WeatherGFM 借鉴 LLM 与视觉基础模型（[[mae|MAE]]、Painter）的 in-context learning 范式，将"是否能用一个通用模型处理多样天气任务与数据模态"作为核心问题[^src-weathergfm]。

## 统一任务表示

所有天气理解任务被抽象为从源数据 $X_S$ 到目标数据 $X_T$ 的投影：

$$\tau: X_S \to X_T$$

- 空间超分：$X_S = x_{LR}$，$X_T = x_{HR}$[^src-weathergfm]
- 天气预报：$X_S = \{x_1,\dots,x_t\}$，$X_T = \{x_{t+1},\dots\}$[^src-weathergfm]
- 图像翻译：将一种模态（如卫星）映射到另一种模态（如雷达）[^src-weathergfm]

## 架构

```
Task-specific Patch Embedding (per task, 处理可变通道数)
    ↓  MLP 对齐到统一嵌入空间
Block-wise Masking (target, 75% mask ratio)
    ↓
Vanilla ViT (MHSA + MLP, pre-LN + residual)
    ↓
Prediction Head (1-layer MLP, hidden 1024) → Unpatchify
    ↓
Reconstructed Target
```

- 主干为**纯 ViT**（borrow 自 Better plain ViT baselines），非层级结构[^src-weathergfm]。
- 输入统一 resize 到 256×256，按 prompt 格式拼接为 $N\times256\times256$[^src-weathergfm]。
- Base：110M 参数（encoder dim 768, depth 12）；Large：330M（encoder dim 1024, depth 24），patch size 16，掩码比例 75%[^src-weathergfm]。
- 训练：L1 loss，AdamW + cosine 调度，base lr 1e-4，16× A100，50 epochs，fp16[^src-weathergfm]。

## 关键技术

- **[[weather-prompt|天气提示设计]]**：三种提示格式分别对应单模态、多模态（跨通道，如 IR069+IR107）、时序输入，将不同模态统一进 in-context 框架[^src-weathergfm]。
- **[[mixed-modal-masked-image-modeling|混合模态掩码图像建模 (MMIM)]]**：把天气理解统一为视觉提问-回答（VQA）；训练时掩码 prompt target 与 ground-truth target、保留 prompt input 与 query input，推理时完全掩码目标即可生成[^src-weathergfm]。
- **任务特定 patch embedding**：解决气象数据物理变量数随数据集/任务变化的问题（区别于 RGB 固定通道）[^src-weathergfm]。

## 任务与数据

- **SEVIR**（Storm Event ImageRy）：GOES-16 三通道（C02/C09/C13）+ NEXRAD VIL + GLM 闪电，11,508 事件、四模态，训练集约 2.2M 图像。涵盖雷达/卫星外推、空间/时序超分、IR↔IR/IR→Vis/IR→Radar 翻译、去模糊后处理[^src-weathergfm]。
- **POMINO-TROPOMI + GEOS-CF**：环境监测翻译任务 GEOS2POES-NO₂（地球静止→极轨卫星 NO₂）[^src-weathergfm]。
- **ERA5**（可扩展性）：48 个 ECMWF 变量预测 T2m，7 个 lead time（6h–7d）单模型一次训练[^src-weathergfm]。

## 关键结果

- 论文报告：在 SEVIR 与 POMINO-TROPOMI/GEOS-CF 设置下，单一 WeatherGFM 在 10 个任务上多数优于各任务单训的 UNet/ViT 单任务模型，并据此认为统一建模可突破单任务性能上限[^src-weathergfm]。
- **Scaling law**：论文报告数据与模型规模增大普遍提升性能（雷达超分尤其需要同时放大数据和模型）；小模型增大数据可能因任务特异性而变差[^src-weathergfm]。
- **OOD 泛化**：论文报告，对接近训练分布的 IR107 外推、IR107→IR069 翻译、15min 时序超分可直接泛化；对差异大的多卫星空间超分失败[^src-weathergfm]。
- **ERA5**：论文报告，单模型在 120h/168h T2m 预报上优于 ClimaX 与 ECMWF IFS，收敛更快（20 epochs/8 A100 vs ClimaX 100 epochs/80 V100）[^src-weathergfm]。
- **Prompt 敏感性**：论文报告，超分任务对 prompt 随机性不敏感；预报与翻译任务波动较大（CSI 标准差约 0.02），searched/high-quality prompt 优于随机 prompt[^src-weathergfm]。

## 局限

- 仅判别式 MSE/L1 训练，无概率/集合预报与不确定性量化[^src-weathergfm]。
- 作者明确不追求每个任务的 SOTA，重点在通用性与 in-context 泛化[^src-weathergfm]。
- OOD 泛化对与训练分布差异大的任务失效[^src-weathergfm]。

## 与相关模型的对比

> [!note] 本课程对照
> 下表及随后互补段为本课程按各论文自述整理的跨论文对照，非论文原文结论。

| 维度 | WeatherGFM | [[uniextreme\|UniExtreme]] | [[swift\|Swift]] | ClimaX / Aurora |
|------|-----------|------------|--------|-----------------|
| 领域 | 天气通用理解（10+ 任务） | 极端天气预测 | 概率天气预报 | 预报 / 大气化学 |
| 统一机制 | 视觉 in-context (prompt) | 频域调制 + 事件先验 | 自回归一致性 | 预训练-微调 / LoRA |
| 多模态 | 是（卫星/雷达/站点） | 否 | 否 | 部分 |
| 跨任务 | 是（单模型多任务） | 否（单任务预测） | 否（仅预报） | 需按任务微调 |
| 生成方式 | 判别式（掩码重建） | 判别式 | 生成式（一致性模型） | 判别式 / 生成式 |
| 未见任务泛化 | 是（OOD prompt） | 否 | 否 | 否 |

（本课程对照）WeatherGFM 与 [[uniextreme|UniExtreme]] 互补：前者横向统一多任务/多模态，后者纵向深耕极端事件预测；二者均不做概率预报，而 [[swift|Swift]] 则专注生成式概率集合预报。它是 [[spatio-temporal-foundation-model|时空基础模型]]在天气领域中"任务统一 + in-context"路线的代表，区别于交通/城市领域的 [[unist|UniST]]、[[urbandit|UrbanDiT]] 等。

## 相关页面

- [[source-weathergfm]] — 源文件摘要
- [[weather-prompt]] — 天气提示设计（三模态格式）
- [[mixed-modal-masked-image-modeling]] — MMIM 训练-推理范式
- [[weather-foundation-model]] — 天气基础模型概念
- [[extreme-weather-forecasting]] — 极端天气预测（互补任务）
- [[uniextreme]] — UniExtreme，极端天气基础模型
- [[swift]] — Swift，自回归一致性概率天气预报
- [[spatio-temporal-foundation-model]] — 时空基础模型范式
- [[mae]] — MAE，掩码图像建模的来源
- [[in-context-learning]] — in-context learning 范式
- [[weatherpeft]] — WeatherPEFT，WFM 参数高效微调框架
- [[task-adaptive-dynamic-prompting]] — TADP，任务自适应软提示（与天气提示对照）

[^src-weathergfm]: [[source-weathergfm]]
