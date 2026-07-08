---
title: "Spatio-Temporal LLM: Reasoning about Environments and Actions"
type: source-summary
tags:
  - multimodal-llm
  - spatiotemporal
  - 2025
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# Spatio-Temporal LLM: Reasoning about Environments and Actions

> Haozhen Zheng, Beitong Tian, Mingyuan Wu, Zhenggang Tang, Klara Nahrstedt, Alex Schwing (UIUC). arXiv:2507.05258, 2025.

该论文提出 **STLLM**（Spatio-Temporal LLM）框架，旨在解决多模态大语言模型（MLLM）在**同时理解全局 3D 环境（点云）和局部时序动态（第一人称视频）** 方面的空缺。核心贡献包括 REA 数据集构建与两种 STLLM 基线架构。[^src-stllm]

---

## 核心挑战

现有 MLLM 在处理"时空提示"（同时指涉全局 3D 环境与局部动作视频的查询）时表现不佳——在 REA 数据集上，现有模型的最佳准确率仅 23.85%–31.46%。[^src-stllm]

---

## REA 数据集

论文设计了 "Reasoning about Environments and Actions"（REA）数据收集流水线，基于 EPIC-KITCHENS（动作标注）、VISOR（物体分割）和 EPIC-FIELDS（稀疏点云）构建。[^src-stllm]

REA 包含五类任务，每类测试不同的时空推理能力：[^src-stllm]

1. **Relative Direction（相对方向）**：判断物体在连续动作中相对于人的方向变化
2. **Relative Distance（相对距离）**：推理人与物体间距离在时间上的变化趋势
3. **Find My Item（物品定位）**：基于场景理解推断物品位置及到达路径
4. **Furniture Affordance Prediction（家具功能预测）**：预测人下一步将与之交互的家具对象
5. **Action Planning（动作规划）**：预测下一动作并提供导航指令

数据集共有 24,371 条训练样本和 1,757 条验证样本，覆盖 300+ 个标注物体和丰富的长尾动作分布。

---

## STLLM 架构

两种基线架构均基于 LLaVA-Video-Qwen2，扩展以处理 3D 点云信息：[^src-stllm]

1. **STLLM-3D**：直接将点云特征（经 Farthest Point Sampling + MLP 投影）与视频帧特征、文本嵌入拼接送入 LLM 解码器。设计简洁、易于实现，但处理大型场景时 token 开销大。

2. **STLLM-Aligner**：引入跨模态对齐模块，通过可学习查询（learnable queries）对视频和文本特征进行交叉注意力，同时通过交叉注意力融合点云空间上下文，输出紧凑的空间特征表示。更为高效，但不如 STLLM-3D 直观易解释。

两种架构共享 SigLip 视觉编码器和 OpenScene 点云编码器（masked transformer decoder）。

---

## 实验结果

在 REA 测试集上：[^src-stllm]
- **STLLM-Aligner** 达到 41.89%/46.50%（ChatGPT-4o/Gemini 2.0 Flash 评测准确率），显著优于现有 MLLM 基线（最高 31.46%/39.50%）
- **STLLM-3D** 取得 40.94%/46.39%，与 Aligner 性能相当
- 在标准 QA 指标（BLEU-4、METEOR、ROUGE-L、CIDEr）上，STLLM 基线在 REA 微调模型基础上进一步提高

在 SQA3D（场景理解 QA 基准）上的零样本评估中，STLLM 同样取得领先，验证了 REA 数据集的跨任务泛化能力。[^src-stllm]

---

## 与 [[source-st-vision-llm]] 的关系

两者均探索多模态 LLM 在时空任务中的应用，但视角不同：
- **ST-Vision-LLM** 面向**交通预测**，将 2D 栅格交通矩阵渲染为图像输入视觉编码器，本质上属于**视觉回归**范式的扩展。
- **STLLM** 面向**具身智能体的时空问答**，同时理解全局 3D 环境（点云）和局部动作（第一人称视频），属于**多模态推理**范式的探索。

---

## 局限性

- REA 数据集仅覆盖厨房场景（EPIC-KITCHENS），领域泛化性有待验证。[^src-stllm]
- 两种 STLLM 基线各有权衡，在 token 效率和可解释性之间尚无最优解。[^src-stllm]
- 位置编码在 STLLM-Aligner 中提升有限，说明 REA 更强调问答推理而非显式 3D 定位。[^src-stllm]

---

## 相关页面

- [[source-multimodal-spatial-reasoning-survey]] — 多模态空间推理综述

## 引用

[^src-stllm]: [[source-stllm]]
