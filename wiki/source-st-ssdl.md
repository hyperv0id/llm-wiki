---
title: "ST-SSDL: Self-Supervised Deviation Learning for Spatio-Temporal Forecasting"
type: source-summary
tags:
  - spatiotemporal
  - forecasting
  - self-supervised
  - traffic
  - deviation
created: 2026-07-21
last_updated: 2026-07-21
source_count: 0
confidence: low
status: active
---

# ST-SSDL: Self-Supervised Deviation Learning for Spatio-Temporal Forecasting

Gao, Dong, Yong, Fukushima, Taura & Jiang (U Tokyo / Toyota), NeurIPS 2025. Code: [github.com/Jimmy-7664/ST-SSDL](https://github.com/Jimmy-7664/ST-SSDL).

## 核心贡献

ST-SSDL 是首个将**自监督偏差学习**（Self-Supervised Deviation Learning, SSDL）引入时空预测的框架。核心洞察：现有方法忽略了当前观测与历史模式之间的动态偏差——这些偏差往往包含关键的预测信号（如突发事件、政策干预）。ST-SSDL 通过三个组件系统化地建模这些偏差：

1. **历史锚点**（Historical Anchor）：将训练数据按周分割平均，为每个当前输入提供时间对齐的历史参考。
2. **可学习原型**（Learnable Prototypes）：用 M 个原型向量离散化连续潜在空间，每个原型代表一种典型时空模式。
3. **两个自监督目标**：对比损失（triplet loss）增强原型间区分度；偏差损失强制物理空间距离与潜在空间距离之间的**相对距离一致性**（$D_1 > D_2 \Rightarrow \tilde{D}_1 > \tilde{D}_2$）。

## 架构

采用 GCRU（Graph Convolution Recurrent Unit）作为 encoder-decoder 骨干。当前输入 $X^c$ 和历史锚点 $X^a$ 并行编码后，通过 query-prototype 交叉注意力映射到原型空间。拼接 $[H^c, V^c, H^a, V^a]$ 后生成自适应邻接矩阵 $\tilde{A}$，驱动 decoder 预测。总损失：$L = L_{\text{MAE}} + \lambda_{\text{Con}} L_{\text{Con}} + \lambda_{\text{Dev}} L_{\text{Dev}}$。

## 实验

在 6 个基准数据集（METRLA, PEMSBAY, PEMSD7(M), PEMS04/07/08）上全面 SOTA。消融实验验证了对比损失和偏差损失缺一不可。效率上，参数量极低（PEMS08 仅 100K），但因 GCRU 的迭代特性，推理速度存在 trade-off。可视化表明模型能根据偏差程度自适应调节原型分配：低偏差→同原型，中偏差→近邻原型，高偏差→远隔原型。

## 局限与未来

论文给出的未来方向为层次化原型结构以增强适应性。框架依赖周级历史平均作为锚点，对非周期性场景的适用性是潜在局限（属编辑性分析，非论文原文明确声明）。
