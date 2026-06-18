---
title: "Multimodal Spatial Reasoning"
type: concept
tags:
  - mllm
  - spatial-reasoning
  - embodied-ai
  - vision-language
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# Multimodal Spatial Reasoning（多模态空间推理）

## 定义

多模态空间推理（Multimodal Spatial Reasoning）是指从异构输入中推断空间关系、位置和动作，并生成可验证的空间锚定输出的能力。形式上，给定输入 $\mathcal{X} = \{x_{\text{img}}, x_{\text{vid}}, x_{\text{pc}}, x_{\text{aud}}, x_{\text{text}}, \ldots\}$（如 RGB 图像、视频、点云、音频和语言），在指定的参考坐标系（2D/3D/ego/allo）下，模型预测输出 $\mathcal{Y}$，可以是：(i) 文本答案/推理链；(ii) 几何量（边界框、姿态、轨迹）；或 (iii) 具身场景中的可执行动作/计划[^src-2510-25760]。

## 空间推理的十种类型

Zheng et al. 将 MLLM 中的空间推理能力归纳为十种基本类型[^src-2510-25760]：

1. **定位与记忆（Localization & Memory）**：在 2D/3D 中定位对象并追踪其状态随时间的变化
2. **关系与几何（Relation & Geometry）**：推理空间关系（上下左右前后）和度量（距离、角度、面积、体积）
3. **导航与问题求解（Navigation & Problem Solving）**：规划路径、优化动作（最短路线、空间谜题）
4. **模式与视角（Pattern & Perspective）**：检测模式/对称性，跨视角推理
5. **缩放与重调（Scaling & Resizing）**：建模尺寸变化同时保持比例
6. **变换（Transformation）**：应用旋转、平移、缩放并维护关系
7. **语境化（Contextualization）**：在环境语境（如房间 vs. 航天器）中解释位置
8. **3D 模型生成（3D Model Generation）**：从空间线索合成 3D 形状/场景
9. **环境建模（Environmental Modeling）**：构建场景/世界模型用于预测和决策
10. **感知与交互（Sensing & Interaction）**：通过传感器/视觉支持实时空间交互（如 AR）

## 评估维度

评估 MLLM 空间推理需探测准确性、鲁棒性、可解释性和泛化性，涵盖六个关键维度[^src-2510-25760]：

1. **多模态整合（Multimodal Integration）**：测试多样化模态组合（图像、文本、音频、深度/点云、传感器）以评估超越单模态线索的跨模态融合
2. **任务覆盖（Task Coverage）**：包含 VQA、3D 定位、地图导航、具身规划、场景/图像生成，跨越低层和高层推理
3. **过程透明（Process Transparency）**：通过注意力图、中间状态或推理探针追踪决策，揭示空间关系如何被编码和操作
4. **泛化与鲁棒性（Generalization & Robustness）**：评估分布外设置（新布局、未见环境、扰动）以测试适应性
5. **交互/具身测试（Interactive/Embodied Testing）**：测量导航/操作和 AR/VR 的实时性能
6. **基准标准化（Benchmark Standardization）**：提供可复现的测试套件，涵盖受控合成任务和真实场景

## 研究方法论谱系

Zheng et al. 将当前增强 MLLM 空间推理的研究路线沿四个维度组织（Figure 3）[^src-2510-25760]：

- **Test-Time Scaling**：训练免策略——改进提示工程（prompt engineering）、工具辅助推理（tool use）、自一致性投票、多模态搜索（VisuoThink）、检索增强生成（Logic-RAG）
- **Post-Training**：监督微调（SFT）和强化学习（RL），使用空间目标数据集、奖励函数和课程
- **模型架构设计**：输入层面注入空间线索（坐标 tokens、深度图、标记通道）或设计专用空间推理模块（3D 编码器、空间特征融合、场景图关系桥）
- **可解释性**：分析 MLLM 空间推理失败的原因——表征不平衡（视觉嵌入压制位置编码）、注意力偏差（仅有 15-20% 权重视空间关系）、缺乏几何先验

## 与其他推理的区别

| 维度 | 文本推理 | 多模态空间推理 |
|------|---------|---------------|
| 输入模态 | 纯文本 | 图像/视频/点云/音频/文本 |
| 核心挑战 | 逻辑链连贯性 | 几何锚定 + 跨模态对齐 |
| CoT 有效性 | 高 | 有限（需要视觉结构化表示） |
| 评估方式 | 答案正确性 | 空间度量精度 + 过程可解释性 |

文本 CoT 在空间推理中效果有限的关键原因在于：空间推理需要显式建模视觉关系，而文本化推理链无法有效传递度量级几何信息[^src-2510-25760]。

## 相关页面

- [[multimodal-large-language-model|MLLM]] — 多模态大语言模型的基础架构
- [[3d-visual-grounding|3D Visual Grounding]] — 从自然语言定位 3D 场景中的对象
- [[vision-language-action|VLA]] — 视觉-语言-动作模型
- [[vision-language-navigation|VLN]] — 视觉-语言导航
- [[embodied-question-answering|EQA]] — 具身问答
- [[spatio-temporal-reasoning]] — 图结构时序数据上的空间推理（本框架的子领域）
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
