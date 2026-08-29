---
title: "DiffPuter: Empowering Diffusion Models for Missing Data Imputation"
type: source-summary
tags:
  - data-imputation
  - diffusion-models
  - expectation-maximization
  - tabular-data
  - iclr-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# DiffPuter: Empowering Diffusion Models for Missing Data Imputation

**作者:** Hengrui Zhang, Liancheng Fang (UIC), Qitian Wu (Broad Institute), Philip S. Yu (UIC，通讯)
**发表:** ICLR 2025。本地 PDF 为 camera-ready 排版：每页页眉 "Published as a conference paper at ICLR 2025"，首页带 arXiv:2405.20690v2 (24 May 2025) 水印；ICLR 2025 著录已在 PDF 页眉核实
**raw:** `raw/zhang-diffputer-iclr-2025.pdf`
**代码:** github.com/hengruizhang98/DiffPuter（论文脚注 1）

## 核心论点

论文提出 DiffPuter，把 EM 算法与扩散模型结合做缺失数据插补，并将自己定位为首个把扩散生成模型整合进 EM 框架的方法（第 2 节，论文自述）[^src-diffputer]。机制上（第 4 节）：M 步固定缺失值填充，用 VE-SDE 简化版扩散（五层 MLP 去噪器）学习完整数据联合密度，score matching 损失是负对数似然上界（Remark 2），故 M 步近似最大似然估计；E 步固定模型，对观测维走前向加噪、缺失维走反向去噪后按掩码合并（式 5-7，RePaint 式混合），Theorem 1 证明采样精确来自 $p_\theta(x\mid x^{obs})$，多次采样取均值为 EAP 估计（默认 N=10）。离散变量 one-hot 编码、按列标准化、缺失项以列均值初始化（第 4.3 节）[^src-diffputer]。

## 实验结果（作者报告）

9 个表格数据集（California/Letter/Gesture/Magic/Bean/Adult/Default/Shoppers/News；摘要写 "ten" 与正文不一致），MCAR/MAR/MNAR 三种机制，主结果 MCAR 30% 缺失率；基线 16-19 个（正文/题注计数不一）。MCAR in-sample 连续列相对最强基线平均提升 6.94% MAE、4.78% RMSE（图 2 题注）；离散列 Accuracy 平均 62.82 排名 1/19（表 1）；out-of-sample 提升 13.37% MAE、4.43% RMSE（表 6 题注，表中均值行为 13.09%/4.60%）；训练时间与 SOTA 同量级、对应性能提升 8%-25%（表 2）；EM 与其他生成模型组合均不及 DiffPuter（表 3）[^src-diffputer]。

## 局限

论文无独立局限性章节。可从文中观察到：仅评测表格数据；极端缺失率下性能趋向均值插补（图 6）；交替训练使总时长高于 MOT/TDM 等基线（表 2）；数据集与基线数量在摘要、正文、图表题注间口径不一[^src-diffputer]。

[^src-diffputer]: [[source-diffputer]]
