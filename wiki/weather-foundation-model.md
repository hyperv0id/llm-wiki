---
title: "Weather Foundation Model"
type: concept
tags:
  - weather-forecasting
  - foundation-model
  - pretraining
  - generalization
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Weather Foundation Model

**天气基础模型（Weather Foundation Model）** 指在大规模气象数据上预训练、可服务于多种天气与气候任务的通用模型，通常借助预训练-微调或 in-context learning 提升泛化能力[^src-weathergfm]。其兴起源于 NLP/CV 基础模型范式向天气与气候领域的迁移[^src-weathergfm]。

## 动机

地球观测系统由卫星、雷达、地面站等多种设备构成，产生多模态数据；为单一任务/场景设计专用模型复杂且劳动密集，而天气数据因受限于单一场景与观测设备存在内在的数据规模瓶颈[^src-weathergfm]。基础模型范式通过大规模预训练统一处理多任务并向未见任务泛化，被视为突破单任务性能上限的途径[^src-weathergfm]。

## 主要路线

- **预训练-微调预报模型**：FourCastNet（自监督预训练 + 自回归微调）、PanguWeather（3D Earth-specific Transformer）、ClimaX（监督式预训练，灵活适配多种预报任务）主要面向天气预报与降尺度[^src-weathergfm]。
- **LoRA / 适配统一**：Aurora 用 LoRA 统一天气预报与大气化学快速预测，但仍需按任务微调，且未建模多模态多任务[^src-weathergfm]。
- **In-context 任务统一**：[[weathergfm|WeatherGFM]]（ICLR 2025）是首个天气**通用**基础模型，通过 [[weather-prompt|天气提示]] 与 [[mixed-modal-masked-image-modeling|MMIM]] 在单一 ViT 内统一 10+ 种天气理解任务（预报、超分、图像翻译、后处理），并展示对未见任务的 in-context 泛化[^src-weathergfm]。
- **极端事件专用基础模型**：[[uniextreme|UniExtreme]] 聚焦多样化极端天气预测，用频域调制 + 事件先验记忆补足通用预报模型在极端事件上的性能差距。
- **生成式概率预报**：[[swift|Swift]] 以自回归一致性模型做单步概率集合预报，关注不确定性量化。

## 与时空基础模型的关系

天气基础模型是 [[spatio-temporal-foundation-model|时空基础模型]] 在欧氏网格气象数据上的特例：输入多为规则网格而非传感器图，任务涵盖预报、降尺度、翻译、后处理等[^src-weathergfm]。与城市/交通领域的 [[unist|UniST]]、[[urbandit|UrbanDiT]] 等相比，天气基础模型更强调多源观测模态（卫星/雷达/站点）与物理变量的可变通道处理[^src-weathergfm]。

## 相关页面

- [[weathergfm]] — WeatherGFM，首个天气通用基础模型
- [[uniextreme]] — UniExtreme，极端天气基础模型
- [[swift]] — Swift，生成式概率天气预报
- [[extreme-weather-forecasting]] — 极端天气预测
- [[spatio-temporal-foundation-model]] — 时空基础模型范式
- [[weather-prompt]] — 天气提示设计
- [[mixed-modal-masked-image-modeling]] — MMIM 训练范式
- [[source-weathergfm]] — WeatherGFM 源文件摘要

[^src-weathergfm]: [[source-weathergfm]]

