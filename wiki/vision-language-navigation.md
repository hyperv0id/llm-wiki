---
title: "VLN (Vision-and-Language Navigation)"
type: concept
tags:
  - vln
  - embodied-ai
  - navigation
  - mllm
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# VLN（Vision-and-Language Navigation，视觉-语言导航）

## 定义

VLN（Vision-and-Language Navigation）是一种协作式多模态任务：agent 在 3D 环境中遵循人类语言指令进行导航，同时需要在模糊条件下进行上下文沟通[^src-2510-25760]。VLN 涉及四个核心组件——视觉感知、语言理解、决策制定和导航执行——全部依赖强大的空间推理能力。

## 三大子任务

Zheng et al. 将 VLN 的空间推理研究组织为三个递进层次[^src-2510-25760]：

### 1. 视觉环境理解与泛化

agent 必须感知周围环境、预判动作后果、并将感知/决策与语言指令对齐（Figure 9, Table XII）[^src-2510-25760]：

- **NaviLLM** (2024)：利用多视图图像捕获当前位置的所有可达视角，构建任务特定 schema 用于 LLM 动作生成
- **SpatialBot** (2025)：通过深度 API 查询环境几何信息并反馈回模型
- **ConceptGraphs** (2024)：关联多视图上的 2D 基础模型输出，构建开放词汇 3D 场景表示
- **Spartun3D-LLM** (2025)：集成 3D 感知 LLM + 位置空间对齐模块，桥接 3D 视觉与文本
- **g3D-LF** (2025)：多尺度预测 novel views 和 BEV 地图，对齐多尺度特征与多粒度语言表示
- **GSA-VLN** (2025)：agent 动态更新参数，利用长期记忆适应环境和多样化用户指令
- **3D-Mem** (2025)：编码多视图 3D 快照的记忆架构，累积和检索空间信息用于长期感知

### 2. 人类意图解释与指令理解

VLN agent 需正确解释空间表达（"左""上""前"），并发展空间推理能力[^src-2510-25760]：

- **LL3DA**：编码 3D 点云，利用注意力机制聚合来自场景和人类交互的上下文信息
- **AutoSpatial**：层次化两轮 VQA 策略——先全局理解，再细节理解
- **RoboPoint**：专门预测空间 affordance 的 VLM，从关系语言中预测精确的动作点

### 3. 路径规划与导航

LLM 常作为高层规划器（Table XIII）[^src-2510-25760]：

- **NavVLM**：VLM 作为认知核心，通过语义理解引导探索
- **SpatialCoT**：双向空间坐标对齐 + Chain-of-Thought grounding，提升推理准确性和可解释性
- **NavCoT**：参数高效自适应 + 自引导导航，生成连贯推理链
- **FlexVLN**：辅助 MLLM 验证 LLM 生成的引导，确保动作可行性，减少幻觉
- **NavA3**：层次框架——推理 VLM 识别目标区域，指向 VLM 通过空间 affordance 进行细粒度定位
- **TopV-Nav**：通过视觉 prompt 构建自适应顶视图地图，为推理提供结构化空间先验
- **BrainNav**：集成双重地图（坐标+拓扑）和双重方向（相对+绝对），支持实时导航与动态场景更新

## 核心挑战

1. **视觉-语言跨模态对齐**：在陌生视角或领域偏移下保持对齐[^src-2510-25760]
2. **环境记忆与时间追踪**：在导航过程中保持和更新空间信息[^src-2510-25760]
3. **指令歧义消解**：处理模糊或上下文依赖的导航指令[^src-2510-25760]
4. **幻觉缓解**：确保 LLM 生成的导航计划在实际环境中可行（FlexVLN 的策略）[^src-2510-25760]

## 相关页面

- [[multimodal-spatial-reasoning|多模态空间推理]] — 核心概念
- [[multimodal-large-language-model|MLLM]] — 多模态大语言模型
- [[3d-visual-grounding|3D Visual Grounding]] — 3D 视觉定位
- [[vision-language-action|VLA]] — 视觉-语言-动作模型
- [[embodied-question-answering|EQA]] — 具身问答
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
