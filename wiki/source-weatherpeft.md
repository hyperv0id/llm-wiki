---
title: "WeatherPEFT: Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models"
type: source-summary
tags:
  - weather-forecasting
  - foundation-model
  - parameter-efficient-fine-tuning
  - peft
  - prompt-tuning
  - fisher-information
  - iclr-2026
created: 2026-07-25
last_updated: 2026-08-08
source_count: 0
confidence: low
status: active
---

# WeatherPEFT: Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models

**Cao, Lin, Cheng, Liu, Li, Wang, Zheng, Liang, Jin, Qin, Cheng & Fu (2026), ICLR 2026**

## 核心论题

WeatherPEFT 是首个探索天气基础模型（WFMs）高效微调的 PEFT 框架。现有 PEFT 方法（LoRA、DoRA、VPT、AdaptFormer 等）设计于视觉/语言领域，无法应对气象下游任务的三大特有挑战：**变量异质性**（不同物理量间的耦合关系随任务变化）、**分辨率多样化**（从 5.625° 到 0.25° 改变主导物理机制——从静力大尺度到非静力对流尺度）、**时空覆盖差异**（全球 vs 区域）。

## 方法

WeatherPEFT 包含两个协同运作的模块：

### TADP（Task-Adaptive Dynamic Prompting）
前向传播阶段，从编码器嵌入权重中提取任务特定信息。首先通过三个专用适配器（HW-Adapter → V-Adapter → D-Adapter）逐层提取空间/分辨率、变量间、气象特征的内部模式；再通过自注意力建模物理量（V）与空间特征（PhPw）之间的外部耦合，生成任务自适应的软提示 token 注入预训练骨干各层。

### SFAS（Stochastic Fisher-Guided Adaptive Selection）
反向传播阶段，利用 Fisher 信息矩阵量化参数对学习目标的敏感度，结合退火随机分量（线性衰减 Uniform 噪声）稳定早期训练中的噪声干扰，选择 Top-k 最关键的参数进行更新，其余参数冻结以保留预训练知识不变。

## 实验

在 Aurora（Bodnar et al., 2025, 1.3B 参数 3D Swin Transformer U-Net）上评估三个任务：

1. **全球降尺度**（5.625°→1.40625°，68 变量）：WeatherPEFT 以 3.48M 可训练参数实现 RMSE 逼近 Full-Tuning（1239.94M），远超 LoRA/DoRA/AdaptFormer/VPT 等。
2. **集合预报后处理**（ENS-10，10 成员→ERA5）：WeatherPEFT（3.18M）在 Z500 上 CRPS 和 EECRPS 均超越 Full-Tuning，体现了任务迁移能力。
3. **中国区域降水预报**（ERA5-CH 0.25°，TP-6hr，12/24/36h）：WeatherPEFT（3.38M）在所有指标上显著超越包括 SCT、Child-Tuning、SAM 等 task-selective 方法在内的所有 PEFT 基线，且在 ∼4% 参数下超越 Full-Tuning。

## 关键贡献

1. 首个针对 WFM 的 PEFT 框架，系统解决气象任务异质性挑战
2. TADP 从编码器嵌入提取内外模式生成任务自适应提示，使模型在前向传播中感知任务上下文
3. SFAS 以 Fisher 信息 + 退火随机性做 principled 参数选择，平衡效率与知识保留
4. 三任务验证：现有 PEFT 与 Full-Tuning 存在显著差距，WeatherPEFT 弥合了这一差距

## 局限性

- 主要基于 Aurora 单一骨干验证，对其他 WFM（如 Prithvi WxC）仅在附录提供初步结果
- 未显式融入大气物理机制/约束

