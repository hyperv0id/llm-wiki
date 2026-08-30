---
title: "Source: Virtual Nodes Improve Long-term Traffic Prediction"
type: source-summary
tags:
  - traffic-forecasting
  - graph-neural-network
  - over-squashing
  - source-summary
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Source: Virtual Nodes Improve Long-term Traffic Prediction

- raw 文件：`raw/virtual-nodes-zhuang-2025.pdf`（arXiv:2501.10048v1 [cs.LG]，水印 2025-01-17；页脚 "A PREPRINT - JANUARY 20, 2025"，PDF 内无会议排版标识，venue 未在 PDF 内核实）
- 作者：Dingyi Zhuang（MIT Civil and Environmental Engineering）、Xiaoyang Cao（清华大学）、Jinhua Zhao（MIT Urban Studies and Planning）、Shenhao Wang（University of Florida）

## 核心论点

论文将 Virtual Nodes（连接全部真实节点的辅助节点）引入 ST-GNN 长期交通预测，针对 over-squashing（引言、Sec 2.3，概念沿引 Alon & Yahav 2020）：消息传递逐层只聚合邻居，l 跳外信息需 l 层才能到达，并在传播中被压缩进固定尺寸表示[^src-virtual-nodes]。机制上提出 semi-adaptive 邻接矩阵（Sec 4.2）：用反对称公式 Aadapt = ReLU(E1E2ᵀ − E2E1ᵀ) 学习虚拟节点接入权重（单向性依据引自 Wu et al. 2020，即 MTGNN），经阈值 r 剪枝后与距离邻接矩阵 Adist 分块拼接，兼顾地理先验与任务驱动学习；另设纯 adaptive 变体作对照；基座模型为 STGCN（Sec 4.1）[^src-virtual-nodes]。

## 论文报告的实验

LargeST SD 子集（716 节点、17,319 边、5 分钟采样，Table 1/Sec 5.1；训练/测试用 2019-2020 共 35,040 帧，Sec 5.3——按 5 分钟采样两年应为 210,528 帧，该数字论文未解释）。作者报告 12 种邻接配置中 Semi-10 V.N. 在全部 horizon 与平均列的 RMSE/MAPE 最低，75-100 分钟平均 RMSE 42.32、MAPE 0.1735，对距离基线 45.15、0.1827 分别降约 6.27% 与 5.04%（Table 2）；敏感性分析中 adaptive 配置不优于距离基线，semi-adaptive 在 10 个虚拟节点处最优、20 个回落（Fig 6）；虚拟节点连接权重可视化显示高权重集中于交叉口等交通活跃区域（Fig 7）[^src-virtual-nodes]。

## 贡献与局限

贡献：将虚拟节点图改写引入交通预测；半自适应邻接设计；层级敏感性与路网热图可解释性证据。局限（论文自述，Sec 5.3/6）：仅 SD 一个子数据集与 STGCN 一种基座，更大规模与更多模型留作未来工作；实验步长 5-100 分钟，报告的长程改进区间为 75-100 分钟平均[^src-virtual-nodes]。

[^src-virtual-nodes]: [[source-virtual-nodes]]
