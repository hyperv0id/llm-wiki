---
title: "GraphCast: Learning skillful medium-range global weather forecasting"
type: source-summary
tags:
  - weather-forecasting
  - medium-range-forecasting
  - graph-neural-network
  - mlwp
  - era5
  - google-deepmind
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# GraphCast: Learning skillful medium-range global weather forecasting

Lam、Sanchez-Gonzalez、Willson、Wirnsberger、Fortunato、Alet、Ravuri 等 7 人共同一作（Google DeepMind + Google Research）提出 **GraphCast**，一种直接从再分析数据训练的机器学习天气预测（MLWP）方法。raw 文件：`raw/graphcast-lam-2022.pdf`（arXiv:2212.12794v2，2023-08-04；preprint 版式，Science 著录未在 PDF 内核实）[^src-graphcast]。

## 核心论点

以 encode-process-decode 配置的 GNN（36.7M 参数）在多分辨率球面 mesh 上学习 6 小时一步的大气演化，自回归滚动生成 10 天、0.25° 全球预报；单块 Cloud TPU v4 上生成 10 天预报不足 1 分钟（摘要、Supp Sec 3.2）[^src-graphcast]。

## 贡献与证据

作者报告：2018 年留出数据上，1380 个验证目标（69 个变量-层组合 × 20 个时效）中 90.3% RMSE 优于 HRES，显著性检验（p≤0.05）下为 89.9%（Figure 2d）；z500 RMSE skill score 改善约 7%–14%（Figure 2b）；落后集中于平流层，排除 50/100 hPa 后为剩余 1180 目标的 99.7%（正文；平流层定位分析见 Supp Sec 7.2.2）；对 Pangu-Weather 在其 252 个目标上 99.2% 领先（Supp Sec 6）。验证以 ERA5 与 HRES-fc0 分别为两系统真值、仅评估 06z/18z 起报以对齐数据同化窗口（Supp Sec 5.2）；multi-mesh 消融显示多层级边是位势短时效优势的必要结构（Supp Sec 7.3.1/Figure 29）。热带气旋路径（18h–4.75d 配对显著）、大气河流 ivt（短时效约 25%）、极端温度分类（5/10 天）等非专门训练任务亦有改进（Figure 3）；重训至 2020 年数据可提升 2021 年预报技巧（Figure 4）。

## 局限（论文自述）

确定性预报，MSE 目标以空间模糊表达不确定性，未建模集合分布；分辨率（0.25°/37 层/6h）与参数规模受 ERA5 原生分辨率与硬件约束；依赖 NWP 同化的高质量再分析数据；论文定位为对传统 NWP 的补充而非替代（Conclusions）[^src-graphcast]。

## 相关页面

- [[graphcast]] — 技术页
- [[multi-mesh-representation]] — 多分辨率 mesh 表征
- [[spherical-geometry-inductive-bias]] — 球面几何偏置路线对照
- [[weather-foundation-model]] — 天气模型谱系

[^src-graphcast]: [[source-graphcast]]
