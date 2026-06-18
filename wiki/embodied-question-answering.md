---
title: "EQA (Embodied Question Answering)"
type: concept
tags:
  - embodied-ai
  - eqa
  - mllm
  - visual-question-answering
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# EQA（Embodied Question Answering，具身问答）

## 定义

EQA（Embodied Question Answering）由 Das et al. (2018) 首次提出，是具身 AI 和机器人学的核心基准[^src-2510-25760]。在此任务中，agent 接收一个自然语言问题（如"Is there a sofa in the living room?"），必须主动探索环境、收集视觉证据并给出答案。挑战在于将语言锚定到空间感知和推理。

## 关键进展

Zheng et al. 综述了以下代表性方法（Table XIV）[^src-2510-25760]：

- **Majumdar et al. (OpenEQA, 2024)**：开发了开放词汇 EQA 数据集，使用 GPT-4V 等基础模型评估，揭示当前系统在需要对象级和场景级空间理解的查询上表现挣扎。GPT-4V 仅达到人类表现的 ~60%[^src-2510-25760]。
- **Tan et al. (2023)**：引入 3D 场景图作为外部记忆，使模型能跨多轮保留和推理空间布局，显著提升多步 QA 效率
- **Hao et al. (Embosr, 2024)**：在 Embosr 框架内集成 Chain-of-Thought 推理，允许在复杂 3D 场景中进行结构化空间推理
- **Zhao et al. (Embodied-R, 2025)**：将感知与推理解耦——大规模 VLM 负责视觉理解，轻量语言模型经 RL 优化负责推理。引入"慢思考"机制增强空间推理深度和可靠性

## 与标准 VQA 的区别

| 维度 | 标准 VQA | EQA |
|------|---------|-----|
| 输入 | 单张图像 | 3D 环境，需主动探索 |
| 答案来源 | 图像内容直接可见 | 需要多步导航收集证据 |
| 核心能力 | 视觉理解 | 视觉理解 + 空间推理 + 策略性探索 |
| 评估 | 答案正确性 | 答案正确性 + 探索效率 |

## 核心洞察

EQA 任务凸显了语言锚定、视觉感知和空间推理在交互环境中错综复杂的相互作用。最近的进展表明，弥合低级视觉输入与高级任务理解之间的鸿沟，需要结合基础模型的强大感知能力与显式推理机制（场景图、神经程序合成、CoT prompting）[^src-2510-25760]。

## 相关页面

- [[multimodal-spatial-reasoning|多模态空间推理]] — 核心概念
- [[multimodal-large-language-model|MLLM]] — 多模态大语言模型
- [[vision-language-navigation|VLN]] — 视觉-语言导航（EQA 常作为 VLN 的子任务）
- [[vision-language-action|VLA]] — 视觉-语言-动作（EQA 为 VLA 的辅助训练任务）
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
