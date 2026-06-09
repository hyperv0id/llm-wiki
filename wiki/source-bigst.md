---
title: "BigST: Linear Complexity Spatio-Temporal GNN for Traffic Forecasting on Large-Scale Road Networks"
type: source-summary
tags:
  - time-series
  - spatio-temporal
  - traffic-forecasting
  - graph-neural-network
  - linear-attention
  - scalability
created: 2026-06-09
last_updated: 2026-06-09
source_count: 1
confidence: medium
status: active
---

# BigST: Linear Complexity Spatio-Temporal GNN for Traffic Forecasting on Large-Scale Road Networks

**作者 / 发表**：Jindong Han、Weijia Zhang、Hao Liu（通讯）、Tao Tao、Naiqiang Tan、Hui Xiong（HKUST / HKUST-GZ / 滴滴）。PVLDB 17(5):1081–1090, 2024，doi:10.14778/3641204.3641217；开源 github.com/usail-hkust/BigST[^src-bigst]。

## 问题
STGNN 是交通预测的主力，但多数需 O(N²) 建模空间依赖、O(T²) 建模长程时间依赖，难以扩展到长历史序列 + 大规模路网[^src-bigst]。为省算力，多数模型只用短窗口（如过去 1 小时），牺牲长程信息；手工周期特征又依赖简单假设[^src-bigst]。[[gwnet|GWNET]] 式自适应邻接 A=σ(E1E2ᵀ) 学图结构需 O(N²)，无法扩展到大路网[^src-bigst]。

## 方法
BigST 把端到端 STGNN 拆成两阶段[^src-bigst]：
- **预处理 — 长序列特征提取器 [[long-sequence-feature-extractor|LSFE]]**：(1) 上下文感知**线性化 Transformer**（借 Performer 正随机特征 PRF 近似 softmax 核 → O(T_l)），生成式预训练编码长程时间动态；(2) 免训练**周期特征采样**（取过去 D 天 / W 周同期特征）。LSFE 输出可**整库预计算缓存**，大幅降低预测阶段开销[^src-bigst]。
- **预测 — 线性化全局空间卷积网络 (LGSCN)**：(1) **Patch 级动态图学习 (PDGL)** 用静态+动态节点嵌入算注意力分数构造时变邻接（温度 τ）；(2) **[[linearized-spatial-convolution|线性化空间卷积 (LSC)]]** 用同一 PRF 核分解 A≈D⁻¹φ(E1)φ(E2)ᵀ，免显式计算稠密邻接，把图卷积降到 **O(N)**[^src-bigst]。末端 concat(LGSCN 输出 ‖ 预计算特征) → MLP **非自回归**出预测[^src-bigst]。

## 结果
在 California（9,638 节点）与 Beijing（99,716 节点，滴滴 GPS）两个大规模数据集上，BigST 全面超越 [[dcrnn|DCRNN]]/ASTGCN/[[gwnet|GWNET]]/AGCRN/STGODE/DSTAGNN：California 平均 MAE/RMSE/MAPE 较最优基线提升 6.3/7.6/8.4%，Beijing 提升 9.3/4.2/3.9%[^src-bigst]。效率上较 GWNET 训练加速 2.3–20.6×、推理 1.7–26.5×、省显存达 76.1%；可扩展到约 **10 万节点**，比常用数据集大两个数量级[^src-bigst]。

## 局限
线性化（PRF 近似、空间卷积近似）带来可感知精度损失（消融证实）；周期采样依赖规则周期性（弱周期的北京更靠长程表征）；仅考虑固定节点集，未处理动态增删节点[^src-bigst]。

[^src-bigst]: [[source-bigst]]
