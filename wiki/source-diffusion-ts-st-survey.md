---
title: "A Survey on Diffusion Models for Time Series and Spatio-Temporal Data"
type: source-summary
tags:
  - diffusion-model
  - time-series
  - spatio-temporal
  - survey
  - generative-model
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# A Survey on Diffusion Models for Time Series and Spatio-Temporal Data

**Authors**: Yiyuan Yang (Oxford), Ming Jin (Griffith), Haomin Wen (CMU), Chaoli Zhang (Zhejiang Normal), Yuxuan Liang (HKUST-GZ), Lintao Ma (Ant Group), Yi Wang (HKU) 等
**Venue**: arXiv:2404.18886（2024，v5 更新至 2025-12，投 ACM Computing Surveys）
**PDF**: raw/diffusion-ts-st-survey.pdf

## 定位

首篇系统覆盖 time series 与 spatio-temporal 双数据的 diffusion model 综述（此前 image/audio 扩散综述不含时序场景）。Fig. 3 时间线显示该领域始于 WaveGrad (2020)，截至 2024 年 Table 1 汇总了 60 个代表性方法[^src-diffusion-ts-st-survey]。

## 分类体系（四维度）

1. **模型类别**：unconditional vs conditional 两大类；细分五型：DDPM（离散马尔可夫链）、score-based（连续时间 SDE，逆过程用 Anderson 1982 反向 SDE 求解）、conditional（如 CSDI 2021、PriSTI 2023）、LDM（如 LDT 2024）、DDIM 加速采样[^src-diffusion-ts-st-survey]。
2. **任务**：四大任务 forecasting、generation、imputation、anomaly detection，Table 1 另标注 classification、denoising、alignment 等次要任务[^src-diffusion-ts-st-survey]。
3. **数据模态**：univariate / multivariate time series / spatio-temporal；ST 数据再分 ST graph（domain-oriented vs domain-agnostic）与 trajectory[^src-diffusion-ts-st-survey]。
4. **应用领域**：8 个——healthcare、traffic、sequential recommendation、climate/weather、energy、audio、AIOps、finance[^src-diffusion-ts-st-survey]。

## 关键机制与覆盖方法

Sec 4.3 指出 U-Net 为去噪网络主流，讨论注入 temporal inductive bias 的设计，并提出 denoiser-as-prior 框架（连接 generative sampling 与 Plug-and-Play 式经典信号恢复）[^src-diffusion-ts-st-survey]。覆盖方法含已入库页面 TimeGrad (2021)、CSDI (2021)、TSDiff (2023)、DYffusion (2023)、DiffSTG (2023)、USTD (2024)，以及 ScoreGrad (2021)、SSSD (2022)、DiffTraj (2024) 等[^src-diffusion-ts-st-survey]。

## 与 wiki 已有综述的差异

[[source-deep-time-series-survey]] 按预测架构（RNN/GNN/Transformer）分类且限于 forecasting；[[source-mts-imputation-survey]] 双视角（插补不确定性 × 架构）仅限 imputation；本综述以生成模型类型为主轴，任务横跨四大类，是 diffusion 视角的补位。

## 局限

覆盖截至 2024 年，未含 consistency model 在时序的实测对比；汇总表只给方法-任务-数据标注矩阵，无统一指标 benchmark 数字[^src-diffusion-ts-st-survey]。

[^src-diffusion-ts-st-survey]: [[source-diffusion-ts-st-survey]]
