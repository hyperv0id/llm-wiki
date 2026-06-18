---
title: "VLA (Vision-Language-Action)"
type: concept
tags:
  - vla
  - embodied-ai
  - robotics
  - mllm
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# VLA（Vision-Language-Action，视觉-语言-动作）

## 定义

VLA（Vision-Language-Action）模型从多模态输入（典型为视觉观察和语言指令）生成可执行动作，使用视觉-语言基础模型作为 backbone[^src-2510-25760]。这些模型是连接感知与物理交互的核心桥梁，被视为通向具身 AGI 的关键路径。

## 研究路线

Zheng et al. 将增强 VLA 空间理解的研究归纳为四个方向[^src-2510-25760]：

### 1. 空间信息输入模态

通过引入深度图和 3D 点云等补充 2D 视觉所缺失的几何线索（Table IX）[^src-2510-25760]：

- **3D-VLA**：引入对象、位置、场景和动作的交互 tokens，将语言模型与生成目标图像/深度图/点云的扩散模型对齐
- **PointVLA**：组合 2D 图像特征（VLM）+ 3D 点云特征（点编码器）输入动作专家
- **SpatialVLA**：从单目深度预测中推导 3D 感知位置编码
- **BridgeVLA**：双阶段训练——预训练 VLM 用于 2D 热力图目标定位 + 微调以多视图正射投影 3D 点云生成动作轨迹

关键局限：大规模 3D 具身数据集的稀缺性；在 2D 视觉-语言数据上大规模训练的模型（如 Gemini Robotics、π₀.₅）仍总体领先[^src-2510-25760]。

### 2. 多任务预训练和联合训练

通过共享表征的辅助任务隐式鼓励空间推理（Table X）[^src-2510-25760]：

- **RT-2**：首次在共享 token 空间中联合训练 VLM 完成 VQA + 机器人动作预测
- **Gemini Robotics**：两阶段——基础 VLM 预训练于轨迹预测、多视图对应、3D 边界框检测，然后微调动作解码器输出低级控制
- **π₀.₅**：预训练 VLM backbone 于 VQA、目标定位、子任务预测和离散动作生成；后训练增加连续控制动作头
- **ChatVLA**：两阶段课程——先从机器人数据学习控制，再逐步引入 VQA 等任务保持与预训练 VLM 的对齐；使用 MoE 架构避免任务干扰

### 3. 显式推理

在动作生成过程中引入结构化中间表示和多步推理[^src-2510-25760]：

- **ECoT**：训练 VLA 模型生成逐步推理链（高层计划、子任务、目标位置、低层动作）
- **Chat-VLA2**：在 ChatVLA 基础上增加 reasoning-following 模块，将生成的动作与 backbone 的内部推理对齐
- **Chain-of-Affordance**：将任务分解为四阶段——识别目标→选择抓取点→定位放置区域→规划轨迹
- **RT-Affordance**：分层 VLA——affordance 预测模型首先生成关键姿态，然后引导反应式 VLA 生成低级控制

### 4. Backbone 空间推理能力评估

Zheng et al. 在多个具身基准上测试了 VLA 中使用的 VLM backbone（Table XI），发现它们即使未经机器人数据微调也展现一定空间推理能力[^src-2510-25760]。详见 [[multimodal-large-language-model#VLA 中 MLLM 的空间推理能力|MLLM 页面]].

## 端到端 vs. 模块化

| 范式 | 代表工作 | 特点 |
|------|---------|------|
| 端到端 | OpenVLA, π₀ | 直接从大规模演示训练 VLM 作为反应式策略，预测低级控制 |
| 模块化 | π₀.₅, ChatVLA | 分解为自然语言子任务 → 反应式控制器或低级 VLA |
| 中间阶段 | Chain-of-Affordance | 插入 affordance/目标状态预测 → 运动规划 → 动作 |

无论控制表示形式如何，空间推理始终是这些系统的核心[^src-2510-25760]。

## 相关页面

- [[multimodal-spatial-reasoning|多模态空间推理]] — 核心概念
- [[multimodal-large-language-model|MLLM]] — 多模态大语言模型
- [[vision-language-navigation|VLN]] — 视觉-语言导航
- [[embodied-question-answering|EQA]] — 具身问答
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
