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

**Authors**: Yiyuan Yang (Oxford) 等 12 人（Griffith、CMU、HKUST-GZ、HKU、Salesforce、ECNU、Fudan、Squirrel Ai 等）
**Venue**: arXiv:2404.18886v5（cs.LG，2025-12-06 更新；版权页标注 ACM Computing Surveys 投稿格式）

## 定位

扩散模型已被用于 time series 与 spatio-temporal 数据的生成、推断与下游任务（healthcare、recommendation、climate、energy、audio、traffic 等领域），本综述沿四个维度组织文献：模型类别、任务类型、数据模态、应用领域，并开源了持续维护的论文仓库。

## 分类体系（四维度）

1. **模型类别**：Sec 4 先讲标准扩散模型（DDPM 与 score SDE），再讲改进型（conditional diffusion、LDM、DDIM 等），最后单独讨论 denoiser；无监督 unconditional 与带标签的 conditioned 两大组贯穿全文，unconditional 内再分 probability-based 与 score-based。
2. **任务**：forecasting、generation、anomaly detection、imputation 四大任务对应 Sec 5.1–5.4；Table 1 的 Task 列另标注 classification（如 CARD 2022）、denoising（如 SGMSE）、alignment（STPP）等次要任务。
3. **数据模态**：univariate / multivariate time series 与 spatio-temporal；ST 数据再分 Spatio-Temporal Graph（含 domain-oriented 与 domain-agnostic）与 Spatio-Temporal Trajectory 两类。
4. **应用领域**：Sec 7 分 8 节——healthcare、traffic、sequential recommendation、climate/weather、energy、audio、AIOps、finance。

## 关键机制与覆盖方法

Sec 4.3 专论 denoiser：U-Net 是扩散模型最常用的去噪网络架构，时序去噪器需注入 temporal inductive bias（如 Temporal U-Net/WaveGrad 的多尺度下采样、DiffSTG/DiffUFlow 的图卷积结构先验）；4.3.3 提出 denoiser-as-prior 框架——直接用 denoiser（或 score function）作为学习到的先验求解逆问题，ImDiffusion、SADI 仅用去噪器恢复缺失段，免去迭代生成采样。Table 1 收录 60 个方法，Year 列最早为 2021（WaveGrad、DiffWave 两行，均 ICLR 2021；其参考文献 [31][115] 为 2020 arXiv 预印本）；论文正文指出 ST 数据上的 DDPM 工作"主要从 2023 年起步"。覆盖的已入库方法：TimeGrad (2021)、CSDI (2021)、SSSD (2023, TMLR)、TSDiff (2024, NeurIPS 36，综述如此标注)、DYffusion (2023)、DiffSTG (2023)、USTD (2023, arXiv:2310.17360，综述如此标注)。

## 局限（论文自述）

采样迭代慢、高维输入带来计算开销，推广到高维实际数据仍是挑战；未来方向一节承认扩散模型计算开销大，难以用于资源受限场景。Table 1 只给方法-任务-数据-应用标注矩阵，不含统一指标下的定量对比。

[^src-diffusion-ts-st-survey]: [[source-diffusion-ts-st-survey]]
