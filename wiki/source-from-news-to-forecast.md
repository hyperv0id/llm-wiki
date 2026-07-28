---
title: "From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection"
type: source-summary
tags:
  - llm-agent
  - spatiotemporal
  - reasoning
  - 2024
created: 2026-07-07
last_updated: 2026-07-28
source_count: 2
confidence: medium
status: active
---

# From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection

> Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, Junhua Zhao. NeurIPS 2024. arXiv:2409.17515.

该论文提出了一种融合**新闻事件文本**与数值时序数据的 LLM 预测框架，核心创新在于利用 LLM **生成式智能体（Generative Agent）** 的推理能力对新闻进行迭代筛选与反思，从而将非结构化文本信息引入时间序列预测。[^src-from-news-to-forecast]

---

## 核心思路

传统时间序列预测仅依赖历史数值数据，无法感知突发事件（如政策变化、自然灾害、社会事件）的外部冲击。该工作提出：

1. **推理智能体（Reasoning Agent）**：基于 LLM（GPT-4 Turbo）的智能体，通过 CoT 提示将新闻分类为长期影响（long-term）和短期影响（short-term），并给出影响合理性分析（Rationality）。该智能体自动或在初始逻辑引导下过滤无关新闻，输出结构化 JSON 供下游使用。[^src-from-news-to-forecast]

2. **评估智能体（Evaluation Agent）**：在模型训练和验证后，评估智能体将预测误差与全部可用新闻进行对比分析，识别是否遗漏了关键新闻事件。识别出的遗漏新闻用于更新推理智能体的过滤逻辑，形成迭代闭环（见下图）。通常 2 次迭代即可显著提升效果。[^src-from-news-to-forecast]

3. **LLM 微调预测**：使用 LoRA 对 LLaMA 系列 LLM 进行指令微调。输入包含历史时序数据（以数字 token 序列表示）、额外辅助信息（天气、日期、经济指标）以及筛选后的新闻文本摘要。输出为未来时序数值的 token 序列，以自回归方式生成预测。[^src-from-news-to-forecast]

---

## 技术要点

- **新闻-时序对齐**：按时间频率、预测展望期和地理区域将新闻与预测任务配对，确保地域与时间相关性。[^src-from-news-to-forecast]
- **多轮迭代过滤**：推理智能体 → 数据配对 → LLM 微调 → 评估智能体 → 逻辑更新 → 下一轮迭代。迭代过程使新闻选择逻辑逐步完善，预测精度逐次提升。[^src-from-news-to-forecast]
- **数据来源**：GDELT 全球事件数据库、News Corp Australia、Yahoo Finance；辅助信息来自 OpenWeatherMap、日历库和 pandas_datareader 经济指标。[^src-from-news-to-forecast]

---

## 实验结果

在 Electricity（澳大利亚各州电力需求）、Exchange（汇率）、Traffic（交通流量）和 Bitcoin（比特币价格）四个领域进行评测：[^src-from-news-to-forecast]

- **过滤新闻显著优于未过滤新闻**：未过滤新闻因引入噪声，在所有指标上**劣于**纯数值基线；而过滤后新闻则全面超越所有基线。
- **跨领域 SOTA 比较**：与 Autoformer、iTransformer、PatchTST、TimesNet、FEDformer 等 10 种主流方法对比，该框架在 MAE/MSE/MAPE 指标上均取得最优或次优。
- **迭代消融**：第一轮选择已超越纯数值基线；第二轮进一步降低误差；第三、四轮在部分领域继续微幅提升。

---

## 与 [[source-exollm]] 的关系

两者都利用 LLM 增强时间序列预测，但切入角度不同：
- ExoLLM（WWW 2025）面向**外生变量**预测场景，通过 Meta-task Instruction + Multi-grained Prompts + Dual TS-Text Attention 将数值型外生序列与 LLM 知识融合。
- 本工作面向**新闻事件文本**与预测的深度融合，强调 LLM 智能体的迭代推理与自我反思能力，而非静态的文本-时序对齐机制。

---

## 局限性

- 依赖 GPT-4 Turbo 作为智能体引擎，推理成本高，且新闻过滤质量受底层 LLM 能力约束。[^src-from-news-to-forecast]
- 新闻收集和预处理流程尚未实现完全自动化输出端到端流水线。
- 实验仅覆盖四个领域，跨领域泛化能力有待验证。

---

同属“外生文本 + 数值”路线；与自建新闻配对不同，[[time-mmd|Time-MMD]] 提供跨 9 域、控污染的公共对齐语料与 MM-TSFlib 试点评测，常被后续多模态方法并列为文本协变量基准[^src-time-mmd]。

## 相关页面

- [[source-gpt4mts]] — 双提示 LLM 文本-数值时序预测
- [[time-mmd]] · [[source-time-mmd]] — 多领域对齐数值–文本基准（对照：本工作自建新闻配对）
- [[multimodal-time-series-forecasting]] — 多模态预测概念

## 引用

[^src-from-news-to-forecast]: [[source-from-news-to-forecast]]
[^src-time-mmd]: [[source-time-mmd]]
