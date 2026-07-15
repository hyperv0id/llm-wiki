---
title: "MiniTraffic"
type: entity
tags:
  - traffic-forecasting
  - pre-training
  - fine-grained
  - lane-level
  - lightweight
created: 2026-07-16
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# MiniTraffic

**MiniTraffic** 是首个专为细粒度交通预测（统一道路级 + 车道级）设计的轻量级预训练模型，仅 ~119K 可训练参数，由复旦大学、香港中文大学、阿里通义实验室和广州大学联合提出，发表于 ICML 2026[^src-minitraffic]。

## 设计动机

细粒度交通预测面临三个瓶颈：(1) 道路数据丰富但车道标注稀缺，数据不平衡阻碍预训练；(2) 道路-车道多粒度统一建模需求；(3) 大型预训练模型（[[urbangpt|UrbanGPT]]、[[unist|UniST]] 等）部署成本过高，在需要频繁重训的细粒度场景中不实用[^src-minitraffic]。

## 核心架构

MiniTraffic 基于统一预训练 + 粒度感知微调范式：

- **预训练阶段**：多源道路数据经 [[frequency-domain-stability-augmentation|FDA]] 频域增强 → Instance Normalization + Patch 分割 → Random Mask（40%）→ Contrastive Clustering 构建 k-NN 稀疏图 → Fine-grained Graph Attention → Reduction Head 还原维度。损失函数 = 重建损失 + InfoNCE 对比损失[^src-minitraffic]。

- **道路级微调**：FDA 替换为 Extension（复制 D 倍匹配预训练结构），输出经 Pooling 模块（entity-wise 平均）恢复原始分辨率[^src-minitraffic]。

- **车道级微调**：保留 FDA，仅更新 Adaptive Head + Reduction Head，支持 few-shot 泛化[^src-minitraffic]。

## 关键性能

在 PeMS-Lane、PeMS-Road、PeMSF-Lane/Road、HuaNan-Lane/Road 六个数据集上与 29 个基线对比[^src-minitraffic]：

| 场景 | 最佳基线 | MiniTraffic 提升 |
|------|---------|-----------------|
| PeMS-Lane (h=6) | McgVAE MAE=4.89 | MAE=3.94 (−19.4%) |
| HuaNan-Lane (h=6) | McgVAE MAE=5.72 | MAE=4.11 (−28.1%) |
| PeMS-Road (h=6) | GPT-ST MAE=3.72 | MAE=3.12 (−16.1%) |

推理延迟比 [[gpt-st|GPT-ST]] 降低 40%+，FLOPs 降低约 85%[^src-minitraffic]。

## 与相关模型的关系

- vs [[gpt-st|GPT-ST]]（NeurIPS 2023）：GPT-ST 是通用 ST 预训练框架，MiniTraffic 专攻细粒度场景。MiniTraffic 参数量仅 119K（GPT-ST ≈ 1,126K），且特有 FDA + Contrastive Clustering[^src-minitraffic]。
- vs [[mcgvae|McgVAE]]（CIKM 2024）：McgVAE 是首个道路-车道联合建模模型，但采用集成架构，无预训练机制[^src-minitraffic]。
- vs FlashST（ICML 2024）：FlashST 是 prompt-tuning 框架，侧重通用迁移；MiniTraffic 专为细粒度设计，参数更少[^src-minitraffic]。

[^src-minitraffic]: [[source-minitraffic]]
