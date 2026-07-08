---
title: "Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review"
type: source-summary
tags:
  - survey
  - foundation-model
  - spatiotemporal
  - multimodal
  - 2026
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# STFM Pipeline Review

**Authors:** Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, Kai Zheng

**Year:** 2025/2026 | **arXiv:** 2506.01364 | **Venue:** IEEE TKDE (under review)

## 核心贡献

本文首次从 Pipeline 视角系统综述了时空基础模型（Spatio-Temporal Foundation Models, STFMs），将 STFM 生命周期分解为数据协调、模型设计（原始/迁移）、训练目标和迁移适配四个阶段，提出创新的数据属性分类法（data property taxonomy）。[^src-stfm-pipeline-review]

## Pipeline 框架

**第一阶段：数据协调（Data Harmonization）**
覆盖五类时空数据：轨迹、事件、时空网格、视频、时空图。详细讨论预处理（过滤、标准化、解耦）、嵌入（空间/时间/频率/谱嵌入）和侧信息（外生变量、多模态、检索）三个子环节。首次系统梳理了各类数据的预处理流程和嵌入技术选择。[^src-stfm-pipeline-review]

**第二阶段：模型设计（Model Design）**
基于数据属性的分类法将 STFMs 分为原始模型（primitive）和迁移模型（transferred）两类。原始模型按数据依赖关系分为时序模型（RNN/Transformer）、空间模型（GNN/FNO）和时空模型（混合架构）。迁移模型分为视觉（ViT/ResNet）、语言（BERT/LLaMA）和多模态（CLIP/BLIP）三类，对应不同的预训练源和适配策略。[^src-stfm-pipeline-review]

**第三阶段：训练目标**
原始模型的训练目标包括回归建模、掩码建模（MAE/MVM）、对比学习和扩散生成。迁移模型的适配技术包括 Prompt Engineering、特征增强（Feature Enhancement）、跨域对齐（Cross-Domain Alignment）和监督微调（SFT）。[^src-stfm-pipeline-review]

**第四阶段：应用**
涵盖交通、天气、能源、金融、医疗和公共服务等领域的实际部署案例。[^src-stfm-pipeline-review]

## 覆盖的模型

综述全面覆盖了主流 STFMs，包括：UniST、UrbanDiT、ClimaX、Moirai、Chronos、TimesFM、STD-MAE、GPT-ST、UrbanGPT、ST-LLM、UniTraj、KGTS、PTR、START 等。对 UniST 的 prompt-empowered 统一预测框架、UrbanDiT 的扩散 Transformer + 统一 prompt 学习、ClimaX 的 variable tokenization + CMIP6 预训练等代表性工作进行深入分析。[^src-stfm-pipeline-review]

## 未来方向

讨论了多目标训练（Multi-Objective Training）、外生变量融合、新型融合架构（cross-attention、MoE、gated mechanisms）、大规模预训练数据构建等开放挑战。强调数据质量与领域相关性比单纯的规模扩展更为重要。[^src-stfm-pipeline-review]

## 交叉链接

- [[source-st-foundation-models-survey]] — 时空基础模型早期综述，本文在 Pipeline 视角和数据属性分类上有所超越
- [[source-climax]] — ClimaX 气象基础模型
- [[source-urbandit]] — UrbanDiT 扩散 Transformer 时空基础模型
- [[source-unist]] — UniST prompt-empowered 时空预测
- [[source-chronos]] — Chronos 时序基础模型
- Moirai — 通用时序基础模型（尚无独立 wiki 页面）
- [[spatio-temporal-foundation-model]] — 时空基础模型概念页面

[^src-stfm-pipeline-review]: [[source-stfm-pipeline-review]]
