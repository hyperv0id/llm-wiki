---
title: "Task-Adaptive Dynamic Prompting"
type: technique
tags:
  - weather-forecasting
  - prompt-tuning
  - parameter-efficient-fine-tuning
  - adapter
  - iclr-2026
created: 2026-07-25
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# Task-Adaptive Dynamic Prompting (TADP)

**Task-Adaptive Dynamic Prompting（TADP）** 是 [[weatherpeft|WeatherPEFT]] 框架的前向传播组件，通过从编码器嵌入权重中提取任务特定信息生成软提示，使预训练骨干在前向传播中感知下游任务的上下文特征[^src-weatherpeft]。

## 设计动机

天气下游任务的编码器嵌入层隐式编码了任务的独特特征——输入变量、分辨率和天气现象信息[^src-weatherpeft]。TADP 的设计目标是显式提取并利用这些信息，而非像通用 PEFT 方法那样对所有输入统一处理。不同于 WeatherGFM 的 [[weather-prompt|in-context weather prompt]]（通过示例对指定任务），TADP 从模型自身的嵌入权重中动态生成提示[^src-weatherpeft]。

## 两步流程

### 1. 内部模式提取（Internal Pattern Extraction）

从嵌入权重 $E \in \mathbb{R}^{D\times V\times P_h\times P_w}$ 出发，以三个专用适配器按递进层级提取[^src-weatherpeft]：

| 适配器 | 处理维度 | 功能 |
|--------|----------|------|
| **HW-Adapter** | $P_h \times P_w$（空间/分辨率） | 学习邻域空间行为与交互模式，建立基础空间上下文 |
| **V-Adapter** | $V$（物理变量数） | 在已细化的空间特征上建模变量间复杂依赖（如温度-湿度耦合） |
| **D-Adapter** | $D$（隐藏维度/气象特征） | 融合前序空间与物理处理，捕获高层通用大气响应模式 |

每个适配器结构相同：LayerNorm → 下投影 → GELU → 上投影。维度通过转置操作 π 在不同适配器间桥接[^src-weatherpeft]。

### 2. 外部模式整合（External Pattern Integration）

将 D-Adapter 输出通过自注意力建模物理量（V）与空间特征（PhPw）之间的外部耦合，随后经线性投影生成最终的软提示 token $E_P \in \mathbb{R}^{P\times D}$（P 为提示长度）[^src-weatherpeft]。这些提示 token 与输入 token 拼接后注入预训练骨干的每个 Transformer block，确保模型在每个计算阶段都在任务特定上下文中处理输入数据[^src-weatherpeft]。

## 消融结果

单独使用 TADP（不含 SFAS）在降尺度实验中 RMSE 显著优于 LoRA/DoRA 等标准 PEFT 方法，但略逊于完整的 WeatherPEFT（TADP+SFAS），表明前向提示与反向参数选择存在协同增益[^src-weatherpeft]。

## 与相关技术的关系

- **[[weather-prompt|WeatherGFM 的 weather prompt]]**：两者机制不同——TADP 从嵌入权重内生生成提示，WeatherGFM 的 prompt 是外部示例对指定任务
- **VPT（Visual Prompt Tuning）**：VPT 学习与任务无关的固定提示向量，TADP 则为每个输入动态生成任务自适应提示
- **通用适配器方法（AdaptFormer, SSF）**：使用固定结构的适配模块，缺乏对气象变量/分辨率异质性的显式建模

## 相关页面

- [[weatherpeft]] — WeatherPEFT 完整框架
- [[stochastic-fisher-guided-adaptive-selection]] — SFAS 反向传播组件
- [[source-weatherpeft]] — 源文件摘要
- [[weather-foundation-model]] — 天气基础模型概念

[^src-weatherpeft]: [[source-weatherpeft]]
