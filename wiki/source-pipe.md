---
title: "PIPE: Physics-Informed Position Encoding for Typhoon Forecasting"
type: source-summary
tags:
  - position-encoding
  - multimodal
  - time-series
  - typhoon
  - satellite-imagery
  - physics-informed
  - vlm
created: 2025-07-14
last_updated: 2025-07-14
source_count: 0
confidence: high
status: active
---

# PIPE: Physics-Informed Position Encoding for Typhoon Forecasting

## 基本信息

- **作者**：Haobo Li, Eunseo Jung, Zixin Chen, Zhaowei Wang, Yueya Wang, Huamin Qu, Alexis Kai Hon Lau（HKUST CSE & ENVR）
- **发表**：NeurIPS 2025
- **代码**：https://github.com/hobolee/PIPE
- **数据集**：Digital Typhoon [22]

## 核心贡献

PIPE 提出了三个主要贡献：

1. **多模态时间序列预测方案**：将卫星图像与数值时间序列联合建模，超越传统单变量/多变量预测范式。
2. **Physics-Informed Positional Encoding (PIPE)**：一种轻量级方法，将物理元数据（时间戳、地理坐标）嵌入 VLM 的位置编码中。包含两个核心创新：
   - 物理知情位置索引（physics-informed positional indexing）：将物理量（年日、小时、经纬度）映射为位置 ID
   - 变频率位置编码（variant-frequency positional encoding）：对不同的物理变量使用不同波长的正弦函数
3. **实验结果**：在 Digital Typhoon 数据集上 SOTA，台风强度预测 MAE 比此前最优的无视觉方法（TiDE）提升 12%。

## 方法要点

PIPE 基于 Qwen-2.5-VL 架构。核心改动仅在位置编码层：将图像 token 的位置 ID 从传统的 3D 索引（temporal + height + width）替换为物理量索引（day-of-year + hour + latitude + longitude），并用变频率正弦函数替代标准正弦位置编码。文本 token 保留标准序列索引和标准正弦编码。图像 token 的位置 ID 被映射到负值范围以避免与文本 token 冲突。位置编码应用 RoPE 后再添加到输入嵌入。

## 实验要点

- **模型规模**：3B / 7B / 32B (LoRA)，4×H800 训练 0.7-3.7 小时（32B LoRA 0.7h / 3B 2.1h / 7B 3.7h）
- **消融核心发现**：视觉数据贡献 8% 强度 MAE 改善，物理知情编码贡献额外 6%
- **泛化性**：在西太平洋训练的模型零样本迁移到澳大利亚区域，性能优于微调后的 Qwen-2.5-VL-3B
- **注意力分析**：PIPE 的注意力集中在台风区域（中心云结构），而 Qwen-2.5-VL 的注意力则分散在整张图像

## 局限性

卫星图像处理增加了计算复杂度；未来工作将探索融入物理定律或约束（如 PDE 约束）以提升可解释性和鲁棒性。
