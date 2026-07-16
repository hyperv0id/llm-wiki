---
title: "ST-SSDL"
type: entity
tags:
  - spatiotemporal
  - forecasting
  - self-supervised
  - traffic
  - gcru
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# ST-SSDL

**ST-SSDL**（Spatio-Temporal Self-Supervised Deviation Learning）是一个结合自监督偏差学习的时空预测框架，由 Gao, Dong, Yong 等人（东京大学 / 丰田）提出，发表于 NeurIPS 2025[^src-st-ssdl]。

## 核心思想

现实时空数据中，当前观测与历史模式之间存在**动态偏差**——例如交通事故导致的流量骤降、节假日出行模式变化。这些偏差不是简单的二值事件，而是随时空上下文连续变化的信号，对预测至关重要却常被现有方法忽略[^src-st-ssdl]。

ST-SSDL 通过三个步骤系统化建模偏差：

1. **历史锚点**：计算周级历史平均作为自监督参考，为每个当前输入提供对齐的历史基线。
2. **原型离散化**：用 M 个可学习原型向量将连续潜在空间划分为离散区域，使偏差可量化。
3. **距离一致性约束**：通过偏差损失强制"物理空间中距离近的样本对在潜在空间中也应相近"，即相对距离一致性原则。

## 架构

- **骨干**：GCRU（Graph Convolution Recurrent Unit）encoder-decoder，堆叠两层。
- **编码**：当前输入 $X^c$ 和历史锚点 $X^a$ 并行通过 GCRU encoder 得到 $H^c, H^a$。
- **原型交互**：Query-prototype 交叉注意力将 $H^c, H^a$ 映射到原型空间，加权求和生成 $V^c, V^a$。
- **自适应图**：拼接 $[H^c, V^c, H^a, V^a]$ 后线性投影 → 自适应邻接矩阵 $\tilde{A}$ 驱动 decoder 预测。
- **联合训练**：$L = L_{\text{MAE}} + \lambda_{\text{Con}} L_{\text{Con}} + \lambda_{\text{Dev}} L_{\text{Dev}}$。

## 性能

在 6 个基准（METRLA, PEMSBAY, PEMSD7(M), PEMS04/07/08）上全面 SOTA。参数量极低（PEMS08 仅 100K，约为 AGCRN（150K）的 66%；论文正文称 AGCRN 为第二轻量，但完整 Table 7 中 STID 117K、GRU 126K 更轻），但因 GCRU 迭代特性推理速度存在 trade-off。

## 相关页面

- [[source-st-ssdl]] — 源文件摘要
- [[ssdl]] — Self-Supervised Deviation Learning 方法详解
- [[spatiotemporal-deviation]] — 时空偏差概念
- [[relative-distance-consistency]] — 相对距离一致性原则
- [[contrastive-learning]] — 对比学习在 SSDL 中的应用
- [[traffic-forecasting]] — 交通预测概览

[^src-st-ssdl]: [[source-st-ssdl]]
