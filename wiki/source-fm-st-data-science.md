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

**Authors**: Yuxuan Liang, Haomin Wen, Yutong Xia, Ming Jin, Bin Yang, Flora Salim, Qingsong Wen, Shirui Pan, Gao Cong（HKUST-GZ、CMU、NUS、Griffith、ECNU、UNSW、NTU 等）
**Venue**: KDD 2025（arXiv:2503.13502，2025-03-12）

## 论文定位

首个覆盖 ST 数据科学全流程的 STFM 综述：此前综述（Jin et al. IJCAI 2023、Jiang & Liang et al. KDD 2024、Zhang et al. 2024、Goodge et al. 2025）只聚焦 data mining 且几乎只讨论 numerical 问题，本文把 data sensing 与 data management 纳入，并补充 inferential 问题[^src-fm-st-data-science]。

## 分类体系

- **模型二分**：STFM = LLM（语言数据上预训练）+ PFM（跨域 ST 数据从头训练）[^src-fm-st-data-science]
- **应用维度**：按 ST 数据科学生命周期三阶段组织——data sensing（citizen reporting 主动感知、Trajectory-LLM 合成轨迹数据）、data management（缺失填补、知识图谱构建、查询检索）、data mining[^src-fm-st-data-science]
- **能力三轴**：perception（时空模式建模，如 UniFlow、UrbanDiT）、optimization（任务适配）、reasoning（对标 DeepSeek-R1，作者指出当前 ST 模型推理能力欠发达）[^src-fm-st-data-science]
- **LLM 用法**：zero-shot 下三种角色——LLM-as-Augmenter（冻结参数注入外部知识）、LLM-as-Predictor（patch & tokenization 弥合模态差）、LLM-as-Agent（领域模型即插即用作工具，可扩展 multi-agent）[^src-fm-st-data-science]
- **PFM 方法论三维**：架构（Transformer / Diffusion / GNN / SSM 如 Mamba4Cast / CNN）；预训练目标（generative / contrastive / hybrid）；数据模态（location、trajectory & event、ST raster、ST graph 四类）[^src-fm-st-data-science]

## 覆盖的模型

按模态：ST raster——Pangu-Weather 2023（39 年 ERA5 数据，3D Earth-Specific Transformer，Nature）、ClimaX 2023（多变量多尺度气候预训练）、UniST 2024（masked pretraining + learnable ST prompt）、FengWu 2023、W-MAE 2023；ST graph——OpenCity 2024（Transformer+GNN 交通预测）；trajectory——TrajFM（轨迹掩码+自回归恢复）、UniTraj 2024（billion-scale 全球轨迹数据集）；event——MOTOR 2024（医疗时序事件）；location——SpaBERT、GeoVectors（OpenStreetMap）、UrbanCLIP 2024（卫星影像城市画像）[^src-fm-st-data-science]。另有 time series 侧 Chronos、Time-MoE、MOIRAI。

## 与 [[source-st-foundation-models-survey]] 的差异

Goodge et al.（A*STAR，2025）提出 4 类泛化能力（domain/spatial/temporal/scale）愿景并实测 6 个模型，但范围限于 mining + numerical 任务；本文按生命周期组织，新增 sensing/management 阶段、inferential 问题，并把数据结构细分到 5 类（location/trajectory/event/raster/graph）。与 [[source-stfm-pipeline-review]]（pipeline 视角讲模型设计与训练）互补：本篇是任务/工作流视角[^src-fm-st-data-science]。

## 未来方向

四条：accuracy vs interpretability 权衡（数值任务上微调 LLM 并不 trivial）；"大模型万能吗"——time series 与 urban planning 中小专家模型在数据充足时胜过 FM，需 hybrid 方案；one-fit-all 全流程 FM（LLM agent 充当 full-stack engineer）；multimodal integration（对齐融合 text/image/video/sensor 的异构源）[^src-fm-st-data-science]。

[^src-fm-st-data-science]: [[source-fm-st-data-science]]
