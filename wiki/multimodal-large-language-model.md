---
title: "MLLM (Multimodal Large Language Model)"
type: entity
tags:
  - mllm
  - vision-language
  - architecture
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# MLLM（Multimodal Large Language Model）

## 定义

MLLM（Multimodal Large Language Model，多模态大语言模型）是一类在多种模态（视觉、音频、文本等）上同时执行感知和推理的大规模神经网络模型[^src-2510-25760]。典型的 MLLM 采用三段式架构（Figure 3）：

1. **预训练 LLM**（如 LLaMA、Vicuna、Qwen）：负责高层语义推理和文本生成
2. **模态编码器**（如 CLIP、DINOv2、SigLIP）：将视觉/音频等输入编码为嵌入向量
3. **连接器/对齐模块**（如 Q-Former、线性投影层、MLP bridge）：将模态编码器的输出对齐到 LLM 的 token 空间

该架构使 MLLM 能够"看到"图像、理解音频，并在统一的语义空间中进行跨模态推理[^src-2510-25760]。

## 代表性模型

Zheng et al. 综述中涉及的主要 MLLM backbone 包括[^src-2510-25760]：

| 模型 | 参数量 | 视觉编码器 | 语言 backbone | 用途 |
|------|--------|-----------|--------------|------|
| LLaVA 系列 | 7B-34B | CLIP-ViT | Vicuna/Mistral | 通用视觉-语言理解 |
| Qwen2-VL | 7B-72B | ViT + 动态分辨率 | Qwen2 | 高分辨率视觉理解 |
| GPT-4V/GPT-4o | 未公开 | 未公开 | GPT-4 | 通用多模态推理 |
| Gemini 系列 | 未公开 | 未公开 | Gemini | 超长上下文多模态 |
| Prismatic | 7B | SigLIP/DINOv2 | LLaMA | VLA backbone (OpenVLA) |
| PaliGemma | 3B | SigLIP | Gemma | VLA backbone (π₀) |
| Phi-3-Vision | 4B | CLIP-ViT | Phi-3 | 轻量级视觉-语言 |

## VLA 中 MLLM 的空间推理能力

Zheng et al. 测试了多个 VLA 中使用的 VLM backbone 在具身空间推理基准上的表现（Table XI）[^src-2510-25760]：

| Benchmark | Prismatic | PaliGemma | Qwen-2-VL | Phi3-Vision |
|-----------|-----------|-----------|-----------|-------------|
| ERQA | 32.25 | 27.25 | 32.50 | 34.00 |
| SpatialEval | 32.13 | 29.86 | 26.80 | **46.46** |
| SPACE | 23.75 | 17.00 | 18.75 | **26.25** |

这些结果表明，即使未经专门的机器人数据微调，VLM backbone 已展现一定的空间推理能力，这也是它们在下游具身任务中表现强劲的原因[^src-2510-25760]。Phi3-Vision 在 SpatialEval 上显著领先（46.46 vs 其他 model 的 26-32），显示轻量模型在精心设计的空间基准上可能优于大模型。

## 空间推理的架构增强

Zheng et al. 总结了三种增强 MLLM 空间推理的架构策略[^src-2510-25760]：

1. **输入层面增强**：
   - SpatialLLM：混合 CLIP（语义）+ DINOv2/MAE（自监督）特征以提升 3D 感知
   - MPDrive：在视频帧上叠加"标记通道"（marker channel）标注物体中心
   - LocVLM：将归一化物体坐标直接附加到文本 prompt 中
   - SpatialBot：同时输入 RGB + 深度图，给 MLLM 伪 3D 视角

2. **专用空间模块**：
   - Spatial-MLLM：引入基于 VGGT 的专用 3D 空间编码器，与 2D 视觉编码器并行
   - SpatialRGPT：插入可插拔深度模块 + 区域级 3D 场景图训练
   - Cambrian-1：引入 SVA（Spatial Vision Aggregator）动态保留高分辨率空间信息
   - Spatial-ORMLLM：在视觉塔内注入 Spatial-Enhanced Feature Fusion block

3. **跨模态对齐**：
   - Q-Former 类模块（LL3DA, GPT4Point, 3UR-LLM）将 3D 特征压缩为语言模型可消费的 token
   - LLaVA 类投影层（Scene-LLM, LLaVA-3D, 3D-LLaVA）使用轻量 MLP 对齐 3D 特征空间

## 核心局限

Zheng et al. 综合多个可解释性研究，诊断出 MLLM 空间推理薄弱的三个根本原因[^src-2510-25760]：

1. **表征不平衡**：视觉嵌入在范数上主导位置编码，导致空间顺序信息被擦除（Qi et al.）
2. **注意力偏差**：注意力权重中仅 15-20% 聚焦于编码空间关系的区域（Chen et al. 的 ADAPTVIS 分析）
3. **缺乏几何归纳偏置**：模型依赖物体共现而非真正的几何锚定（Rajabi et al.），仅靠扩大数据规模无法解决（Zhang et al.）

## 相关页面

- [[multimodal-spatial-reasoning|多模态空间推理]] — 核心概念总览
- [[3d-visual-grounding|3D Visual Grounding]] — 从语言定位 3D 场景中的物体
- [[vision-language-action|VLA]] — 视觉-语言-动作模型
- [[vision-language-navigation|VLN]] — 视觉-语言导航
- [[source-2510-25760|源文件：Zheng et al. Survey (2025)]]

[^src-2510-25760]: [[source-2510-25760]]
