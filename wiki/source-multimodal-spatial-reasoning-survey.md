---
title: "Multimodal Spatial Reasoning Survey"
type: source-summary
tags:
  - spatial-reasoning
  - multimodal
  - survey
  - vision-language
  - embodied-AI
  - 3D-reasoning
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# Multimodal Spatial Reasoning: A Survey

## 概述

这是一篇关于多模态空间推理的综合性综述，系统梳理了从 2D/3D 视觉定位到具身 AI 等领域的空间推理方法、基准和挑战[^src-multimodal-spatial-reasoning-survey]。论文以多模态大语言模型（MLLM）时代为背景，覆盖了空间推理的完整技术栈，从基础 2D 空间关系理解到复杂的 4D 时空推理。

## 2D 空间推理

在 2D 图像空间推理方面，论文综述了空间 VQA（视觉问答）的方法演进[^src-multimodal-spatial-reasoning-survey]。早期方法聚焦于基本的空间关系判断（如物体的相对位置），而 MLLM 时代的进展包括：基于强化学习的空间一致性微调（Spatial-R1、R1-Zero-like training）、3D 重建辅助的隐式空间推理（Zero-1-to-3）、坐标增强的定位策略（LocVLM）、以及基于 marker 的 prompt 学习（MPDrive）。SpatialBot 引入深度 API 查询几何信息，SSR 通过推理解释引导深度感知。Struct2D 和 Image-of-Thought 则探索了结构化 prompt 增强空间推理的路径。

## 3D 视觉定位与场景推理

3D 视觉定位任务要求模型从自然语言描述在 3D 场景中定位目标[^src-multimodal-spatial-reasoning-survey]。方法分为三类：LLM-based agent 方法（LLM-Grounder、VLM-Grounder 使用 LLM 作为 agent 分解任务，SeeGround 引入视觉 API 交互）；LVLM 引导的特征投影（ReasonGrounder 使用层次化特征 splatting）；以及端到端训练方法（3DVG-Transformer、M3DRef-CLIP 通过多模态预训练学习对齐）。

3D 场景推理方法可分为训练需要和训练不需要两类[^src-multimodal-spatial-reasoning-survey]。训练需要方法通过 Q-Former（3D-LLM、LL3DA、GPT4Point）或 LLaVA 风格投影层（LEO、Scene-LLM、LLaVA-3D）对齐 3D 特征与语言模态。训练不需要方法如 SpatialPIN 使用渐进式 prompt 分解 3D 表示，Agent3D-Zero 使用 Set-of-Line 策略选择多视角进行分析。

## 3D 生成中的空间推理

3D 生成要求强空间推理能力[^src-multimodal-spatial-reasoning-survey]。布局生成方面，LayoutGPT 使用程序化推理生成空间布局规范，HOLODECK 引入优化技术确保物理真实性，I-Design 和 Generation Agents 通过多 agent 系统逐步细化。程序化生成方面，3D-GPT 将自然语言转化为 Blender 脚本，CAD-GPT 集成空间 token 和位置嵌入生成 CAD 序列，CAD-Recode 从点云数据逆向工程生成 CadQuery 脚本。

## 具身 AI 中的空间推理

具身 AI 是空间推理的关键应用领域[^src-multimodal-spatial-reasoning-survey]。VLA（视觉-语言-动作）模型通过三种策略增强空间理解：融入空间模态如深度图和点云（3D-VLA、PointVLA、SpatialVLA）；多任务预训练/共训练（Gemini Robotics 联合训练 3D 检测和轨迹预测，π0.5 预训练 VQA 和动作生成）；显式推理链（ECoT 生成逐步推理，Chain-of-Affordance 将任务分解为四阶段 affordance 推理）。

VLN（视觉-语言-导航）涉及环境感知、指令理解和路径规划[^src-multimodal-spatial-reasoning-survey]。ConceptGraphs 构建开放词汇 3D 场景图，NaviLLM 使用多视图图像和 schema 指令，SpatialCoT 引入双向空间坐标对齐和 CoT 接地。其他具身任务包括具身问答（Embosr 集成 CoT 推理）、具身抓取（ThinkGrasp 使用目标驱动 prompt 识别遮挡对象）和具身世界模型（TESSERACT 模拟 3D 环境的时间演化）。

## 视频与音频空间推理

视频空间推理面临动态场景中的空间一致性挑战[^src-multimodal-spatial-reasoning-survey]。Spatial-R1 通过强化学习微调增强空间一致性，Video-R1 使用空间对齐损失保持帧间一致性，ST-Think 引入双模态骨干处理 4D 推理。音频空间推理方面，SpatialSoundQA 是首个大规模空间音频 QA 基准，BAT 模型结合空间音频编码器和课程学习，SAVVY 则整合音频和视觉线索进行 3D 空间推理。

## 关键挑战

当前 MLLM 在空间推理中仍面临重大挑战[^src-multimodal-spatial-reasoning-survey]：token 压缩导致空间细节丢失、缺乏鲁棒的空间记忆机制、3D 数据和标注稀缺、开放世界多声源场景的泛化困难，以及训练 3D 感知模型的高计算成本。论文指出，未来方向包括更丰富的数据收集管线、更专业的模型架构，以及空间感知与语言推理的更紧密整合。

[^src-multimodal-spatial-reasoning-survey]: [[source-multimodal-spatial-reasoning-survey]]