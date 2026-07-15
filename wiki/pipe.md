---
title: "PIPE (Physics-Informed Position Encoding)"
type: entity
tags:
  - position-encoding
  - multimodal
  - time-series
  - typhoon
  - satellite-imagery
  - vlm
  - physics-informed
created: 2025-07-14
last_updated: 2025-07-14
source_count: 1
confidence: medium
status: active
---

# PIPE (Physics-Informed Position Encoding)

PIPE 是 HKUST 团队在 NeurIPS 2025 提出的轻量级方法，将物理元数据（时间戳、地理坐标）嵌入 VLM 的位置编码中，用于多模态台风预测[^src-pipe]。

## 架构

PIPE 基于 Qwen-2.5-VL 架构，改动集中在位置编码层：

1. **物理知情位置索引（Physics-Informed Positional Indexing）**：将图像 token 的位置 ID 从传统 3D 索引（temporal+height+width）替换为物理量索引——年日 $t_{\text{day}} \in [0,365]$、小时 $t_{\text{hour}} \in [0,23]$、纬度 $\text{lat} \in [0,180]$、经度 $\text{lng} \in [0,360]$。图像 token 的位置 ID 被映射到负值以避免与文本 token 冲突[^src-pipe]。

2. **变频率位置编码（Variant-Frequency Positional Encoding）**：针对不同物理变量使用不同波长的正弦函数。时间维度：$p_{\text{day}}=366$、$p_{\text{hour}}=24$；空间维度：$p_{\text{lat}}=180$、$p_{\text{lng}}=360$。文本 token 保留标准正弦位置编码以保持与预训练 LLM 的兼容性[^src-pipe]。

3. **RoPE 应用**：物理知情位置索引的结果通过 RoPE 编码为旋转矩阵，与变频率正弦位置编码相加后融入输入嵌入[^src-pipe]。

## 性能

- 在 Digital Typhoon 数据集上 SOTA，台风强度 MAE 比此前最优的无视觉方法（TiDE）提升约 12%[^src-pipe]
- 视觉数据贡献约 8% 强度 MAE 改善，PIPE 编码额外贡献约 6%[^src-pipe]
- 西太平洋训练模型零样本迁移澳大利亚区域，性能超越微调 Qwen-2.5-VL-3B[^src-pipe]
- 注意力分析显示 PIPE 的注意力集中在台风区域（中心云结构），而非分散在整张图像[^src-pipe]

## 与 PINN 的区别

PIPE 与传统 PINN（损失约束型或架构嵌入型）不同，属于第三种范式：**编码知情型**——通过位置编码将物理知识注入模型，无需额外损失函数或定制架构模块[^src-pipe]。

## 相关页面

- [[source-pipe]] — 论文摘要
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测
- [[rope]] — RoPE 旋转位置编码
- [[physics-informed-neural-network]] — PINN 方法论

[^src-pipe]: [[source-pipe]]
