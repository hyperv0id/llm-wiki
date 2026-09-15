---
title: "Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey"
type: source-summary
tags:
  - spatio-temporal
  - foundation-model
  - survey
  - llm
  - pretraining
  - data-management
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey

**Authors**: Yuxuan Liang, Haomin Wen, Yutong Xia, Ming Jin, Bin Yang, Flora Salim, Qingsong Wen, Shirui Pan, Gao Cong（HKUST-GZ、CMU、NUS、Griffith、ECNU、UNSW、Squirrel Ai Learning、NTU）
**Venue**: arXiv:2503.13502v1（cs.DB，2025-03-12）。正文用 ACM 会议模板但 venue 字段是占位符（"Conference'17"），具体收录会议在该文本中无法验证。

## 论文定位

论文自述"提供首个覆盖 ST 数据科学全流程的综述"（sensing/management/mining 三阶段），并将差距归为两点：此前综述 [32,54,81,169] 主要把 LLM 当 data mining 工具、较少涉及 sensing 与 management；且偏数值问题（预测、填补）而忽略推断型问题如决策系统。Table 1 对比了 Jin et al.（2023）、Jiang et al.（IJCAI-24，注意其正文表格误标 KDD，以参考文献 [48] 为准）、Liang et al.（KDD-24）、Zhang et al.（2024）、Goodge et al.（2025）。

## 分类体系

- **STFM 二分**：LLM（语言数据上预训练，zero/few-shot 使用）与 PFM（跨域 ST 数据从零训练）。
- **生命周期三阶段**：sensing（处理 citizen report、优化 participatory sensing、规模化合成数据）、management（清洗、知识图谱构建、跨模态检索）、mining。
- **能力三轴**：Perception、Optimization、Reasoning（分 common-sense/numerical/causal 三类；作者指现有 ST 模型推理能力相对 DeepSeek-R1 欠发达）。
- **PFM 三维**：架构（Transformer/Diffusion/GNN/SSM/CNN）；预训练目标（generative/contrastive/hybrid）；数据模态五类（location、trajectory、event、ST raster、ST graph）。
- **LLM 零样本三角色**：Augmenter（参数冻结，注入外部知识）、Predictor（prompt engineering 或 patch & tokenization 弥合模态差）、Agent（领域模型即插即用作工具，可扩展 multi-agent）。

## 覆盖的模型

ST raster：Pangu（Nature 2023，39 年全球气候数据，超越主流数值天气预报）、ClimaX（ICML 2023，多变量多尺度气候预训练）、UniST（KDD 2024，masked pretraining + learnable ST prompt）、FengWu（2023 arXiv，中期预报超 10 天 lead）、W-MAE（masked autoencoder 做 ST grid 预测）。ST graph：OpenCity（2024，Transformer+GNN 交通预测）。Trajectory/event：TrajFM（2024，轨迹掩码+自回归恢复，支持 region/task transferability）、UniTraj（2024，billion-scale 轨迹数据集）、MOTOR（2023，医疗结构化记录 time-to-event FM）。Location：SpaBERT（2022）、GeoVectors（2021，基于 OpenStreetMap 学 location embedding）、UrbanCLIP（WebConf 2024，卫星影像城市 region profiling）。时间序列侧：Chronos（2024）、Time-MoE（ICLR 2025，billion-scale MoE）、MOIRAI（2024）、SSM 系 Mamba4Cast（2024，零样本预测）。

## 与其他综述的差异

按 Table 1，被对比的 5 篇综述均只标 mining、应用仅 N（数值）；本文标 N+I，覆盖三阶段与五类数据结构（L,T,E,R,G）。Goodge et al. 在本篇中仅以 mining 类综述出现，本文未展开其细节。与 [[source-st-foundation-models-survey]] 是不同范围、不同组织方式的综述，可对照阅读。

## 未来方向（附录 A，四条）

1. accuracy 与 interpretability 之咒：直接用 LLM 做数值任务 non-trivial。
2. "Large foundation models are all we need?"：time series 与 urban planning 中，领域数据充足时小专家模型常胜过 FM，作者建议 hybrid 方案。
3. one-fit-all 全流程 FM：需 LLM agent 充当 full-stack engineer。
4. 多模态整合：对齐融合 text/image/video/sensor 异构源。

[^src-fm-st-data-science]: [[source-fm-st-data-science]]
