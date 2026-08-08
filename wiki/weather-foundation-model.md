---
title: "Weather Foundation Model"
type: concept
tags:
  - weather-forecasting
  - foundation-model
  - pretraining
  - generalization
  - parameter-efficient-fine-tuning
created: 2026-06-08
last_updated: 2026-08-08
source_count: 4
confidence: medium
status: active
---

# Weather Foundation Model

**天气基础模型（Weather Foundation Model）** 指在大规模气象数据上预训练、可服务于多种天气与气候任务的通用模型，通常借助预训练-微调或 in-context learning 提升泛化能力[^src-weathergfm]。其兴起源于 NLP/CV 基础模型范式向天气与气候领域的迁移[^src-weathergfm]。

## 动机

地球观测系统由卫星、雷达、地面站等多种设备构成，产生多模态数据；为单一任务/场景设计专用模型复杂且劳动密集，而天气数据因受限于单一场景与观测设备存在内在的数据规模瓶颈[^src-weathergfm]。基础模型范式通过大规模预训练统一处理多任务并向未见任务泛化，被视为突破单任务性能上限的途径[^src-weathergfm]。

## 主要路线

- **预训练-微调预报模型**：FourCastNet（自监督预训练 + 自回归微调）、PanguWeather（3D Earth-specific Transformer）、ClimaX（监督式预训练，灵活适配多种预报任务）主要面向天气预报与降尺度[^src-weathergfm]。
- **LoRA / 适配统一**：Aurora 用 LoRA 统一天气预报与大气化学快速预测，但仍需按任务微调，且未建模多模态多任务[^src-weathergfm]。[[weatherpeft|WeatherPEFT]]（ICLR 2026）论文自称是首个针对 WFM 的 PEFT 框架，以 [[task-adaptive-dynamic-prompting|TADP]] + [[stochastic-fisher-guided-adaptive-selection|SFAS]] 实现任务自适应微调，在三类下游任务上以极少参数逼近甚至超越全量微调[^src-weatherpeft]。
- **In-context 任务统一**：[[weathergfm|WeatherGFM]]（ICLR 2025）是首个天气**通用**基础模型，通过 [[weather-prompt|天气提示]] 与 [[mixed-modal-masked-image-modeling|MMIM]] 在单一 ViT 内统一 10+ 种天气理解任务（预报、超分、图像翻译、后处理），并展示对未见任务的 in-context 泛化[^src-weathergfm]。
- **极端事件专用基础模型**：[[uniextreme|UniExtreme]] 聚焦多样化极端天气预测，用频域调制 + 事件先验记忆补足通用预报模型在极端事件上的性能差距。
- **生成式概率预报**：[[swift|Swift]] 以自回归一致性模型做单步概率集合预报，关注不确定性量化。[[climatear|ClimateAR]]（ICML 2026）以 VAR 范式做多尺度自回归概率气候预测，在月尺度上 ACC 提升 37.56%[^src-climatear].
- **几何感知 S2S 预报**：[[cirt|CirT]]（ICLR 2025）以圆形分块 + 傅里叶域自注意力显式编码球面几何偏置，直接预测 2-6 周平均态，超越 ECMWF 等数值系统[^src-cirt]。

## 高效微调

随着 WFM 规模增长（Aurora 1.3B → Prithvi WxC 2.3B），全量微调的计算与存储成本不可持续[^src-weatherpeft]。[[weatherpeft|WeatherPEFT]]（ICLR 2026）提出了首个针对 WFM 的 PEFT 方案，核心创新在于识别天气下游任务的三大特有挑战——变量异质性、分辨率多样性和时空覆盖差异——并通过 [[task-adaptive-dynamic-prompting|TADP]]（前向任务自适应提示）和 [[stochastic-fisher-guided-adaptive-selection|SFAS]]（反向 Fisher 引导参数选择）双阶段协同解决[^src-weatherpeft]。在降尺度、集合后处理和区域降水预报三任务上，WeatherPEFT 以 ∼3-4M 参数（骨干的 ∼0.3%）大幅缩小与 Full-Tuning 的差距；论文报告增大预算至 ∼3-4% 时与 Full-Tuning 持平或更优，而 LoRA/DoRA 等通用 PEFT 方法存在显著性能差距[^src-weatherpeft]。

## 与时空基础模型的关系

天气基础模型是 [[spatio-temporal-foundation-model|时空基础模型]] 在欧氏网格气象数据上的特例：输入多为规则网格而非传感器图，任务涵盖预报、降尺度、翻译、后处理等[^src-weathergfm]。与城市/交通领域的 [[unist|UniST]]、[[urbandit|UrbanDiT]] 等相比，天气基础模型更强调多源观测模态（卫星/雷达/站点）与物理变量的可变通道处理[^src-weathergfm]。而本课程认为，微调效率是天气与城市/交通基础模型共享的部署瓶颈，WeatherPEFT 的任务自适应 PEFT 范式对该领域具有借鉴价值。

## 相关页面

- [[weathergfm]] — WeatherGFM，首个天气通用基础模型
- [[uniextreme]] — UniExtreme，极端天气基础模型
- [[swift]] — Swift，生成式概率天气预报
- [[extreme-weather-forecasting]] — 极端天气预测
- [[cirt]] — CirT，几何感知 S2S Transformer
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测概念
- [[climatear]] — ClimateAR，VAR 自回归概率气候预测
- [[mixed-scale-conditioning]] — 混合尺度条件控制
- [[spatio-temporal-foundation-model]] — 时空基础模型范式
- [[weather-prompt]] — 天气提示设计
- [[mixed-modal-masked-image-modeling]] — MMIM 训练范式
- [[weatherpeft]] — WeatherPEFT，WFM 参数高效微调框架
- [[task-adaptive-dynamic-prompting]] — TADP 技术
- [[stochastic-fisher-guided-adaptive-selection]] — SFAS 技术
- [[source-cirt]] — CirT S2S 预测论文
- [[source-weatherpeft]] — WeatherPEFT 源文件

[^src-weathergfm]: [[source-weathergfm]]
[^src-cirt]: [[source-cirt]]
[^src-climatear]: [[source-climatear]]
[^src-weatherpeft]: [[source-weatherpeft]]
