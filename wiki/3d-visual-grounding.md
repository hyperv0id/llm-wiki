---
title: "3D Visual Grounding"
type: technique
tags:
  - 3d-vision
  - spatial-reasoning
  - mllm
  - visual-grounding
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# 3D Visual Grounding（3D 视觉定位）

## 定义

3D 视觉定位（3D Visual Grounding）是指给定自然语言描述，在 3D 场景中定位目标物体的任务[^src-2510-25760]。该任务需要强大的空间推理能力来处理复杂的自然语言指令，并结合语言理解与 3D 空间推理，是机器人和 AR 的核心能力。

形式化：给定 3D 场景 $\mathcal{S}$ 和语言查询 $\mathcal{Q}$（如"在桌子上的红色杯子"），模型需输出目标物体的 3D 位置（边界框或点云段）。

## 三类方法

Zheng et al. 将基于 MLLM 的 3D grounding 方法按输入数据模态分为三类（Table VI）[^src-2510-25760]：

### 1. 3D 输入（直接使用 3D 表示）

直接将 3D 格式（点云、体素、学习到的体积特征）嵌入 MLLM[^src-2510-25760]：

- **LLM-Grounder** (2023)：采用粗到细策略——MLLM 解析复杂语言概念，开放词汇 3D 视觉模块生成候选提议，然后评估与查询的语义对齐。完全零样本，开放词汇，闭环反馈驱动
- **Grounded 3D-LLM** (2023)：将场景-指称 tokens 集成到 MLLM 中，通过对齐训练使 3D 输入可被消费
- **Vigor** (2024)：使用 LLM 推断实体的指称顺序（referential order），增强细粒度空间推理

挑战：3D 数据结构复杂，阻碍模型可解释性；标注 3D 数据集有限，制约开放世界泛化[^src-2510-25760]。

### 2. 多视图 2D 输入

将 3D 场景渲染为多视图 2D 图像，利用现有 2D MLLM 的空间推理能力[^src-2510-25760]：

- **ViewRefer** (2023)：引入可学习的多视图原型（multi-view prototypes）捕获跨视图关系
- **VLM-Grounder** (2024)：动态拼接图像序列 + grounding 反馈机制
- **3DAxisPrompt** (2025)：在真实世界场景中插入 3D 坐标轴，将 3D 坐标编码到 prompt 中

关键挑战：视角差异（view discrepancy）——模型的 2D 视角与 grounding 指令源之间的错位[^src-2510-25760]。

### 3. 2D+3D 混合输入

结合两种模态的优势[^src-2510-25760]：

- **SpatialRGPT** (2024)：指出仅依赖 RGB 像素的局限性，提出集成相对深度图与 RGB 图像的可插拔深度模块 + 区域级 3D 场景图训练
- **SeeGround** (2025)：动态调整视角以捕获关键细节，整合 2D 视觉与显式 3D 空间描述
- **ReasonGrounder** (2025)：集成 LVLM + 3DGS（3D Gaussian Splatting）+ 层次特征，在遮挡下实现非模态感知（amodal perception）
- **ZSVG3D** (2024)：首次在 3DVG 中使用程序生成（visual program interface），标准化空间关系

## 代表性方法对比

| 方法 | 年份 | 输入 | Backbone | 亮点 |
|------|------|------|----------|------|
| LLM-Grounder | 2023 | 点云 | GPT-4 | 零样本开词汇闭环 grounding agent |
| Grounded 3D-LLM | 2023 | 点云 | Tiny-Vicuna-1B | 用 LLM 统一 3D 任务建模 |
| VLM-Grounder | 2024 | 多视图 | GPT-4V | 动态拼接 + 反馈机制 |
| SpatialRGPT | 2024 | RGB-D | LLaMA2-7B | 模块化设计，灵活集成深度 |
| ReasonGrounder | 2025 | RGB+3DGS | LLaVA 1.5 | 层次特征 + 非模态感知 |

## 核心挑战

1. **MLLM 的全局理解与局部定位的矛盾**：为全局图像理解设计的 MLLM 难以解析特定物体区域[^src-2510-25760]
2. **几何信息缺失**：空间感知需要超越 RGB 的深度或空间坐标等几何信息[^src-2510-25760]
3. **3D 数据稀缺**：高质量标注 3D 数据集的有限性限制了模型的泛化能力[^src-2510-25760]

## 相关页面

- [[multimodal-spatial-reasoning|多模态空间推理]] — 核心概念
- [[multimodal-large-language-model|MLLM]] — 多模态大语言模型
- [[vision-language-navigation|VLN]] — 视觉-语言导航（下游应用）
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
