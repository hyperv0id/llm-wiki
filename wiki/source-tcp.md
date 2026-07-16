---
title: "TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness"
type: source-summary
tags:
  - tropical-cyclone
  - precipitation-forecasting
  - diffusion-model
  - multi-modal
  - nwp-integration
created: 2026-07-24
last_updated: 2026-07-24
source_count: 0
confidence: low
status: active
---

# TCP-Diffusion: 多模态扩散模型用于全球热带气旋降水预测

Huang, Mu, Bai & Watson (ICML 2025) 提出了 TCP-Diffusion，首个基于深度学习的全球热带气旋（TC）降水预测模型。论文发表于 ICML 2025，来自浙江工业大学和布里斯托大学。

## 核心问题

TC 降水造成的灾害损失超过大风，但现有降水预测研究几乎全部聚焦固定区域的常规降水，忽略了随 TC 移动的动态预测目标。直接将常规降水预测方法应用于 TC 降水存在累积误差大、缺乏物理一致性的问题，且未有效利用 TC 相关气象要素和 NWP 预测信息。

## 三大核心贡献

1. **Adjacent Residual Prediction (ARP)**：将训练目标从绝对降水值改为相邻时间步的降水变化量（$\Delta$ Rainfall），通过累积得到最终预测。此机制赋予模型"变化感知"能力，减少累积误差并确保物理一致性。

2. **多模态编码框架**：构建多个专用编码器提取异构气象信息——Historical Data2d Encoder（3D CNN 编码 2D 降水/环境数据）、Historical Data1d Encoder（MLP+Transformer 编码标量 TC 变量如强度、位置）、Future Data2d Encoder（ResNet-18 编码 ERA5-IFS 预测数据）。核心去噪网络 EA-3DUNet 在 3DUNet 基础上添加空间注意力和时间注意力模块。

3. **NWP 集成**：首次将低成本 NWP 预测（ERA5-IFS）作为条件引导 DL 扩散模型，实现"DL 增强低质量 NWP 超越高质量 NWP（ECMWF-IFS）"。

## 关键结果

在 1877 个全球 TC（1980-2020，六个洋区）上训练评估。3h 分辨率预测未来 12h。TCP-Diffusion 在中等/强降水的 ETS 和总降水量 TPM AE 上均达 SOTA，且是唯一在所有指标上超越 Persistence 基线的 DL 模型。与 ECMWF-IFS 对比，ETS-6 达 0.412（ECMWF-IFS 0.302），TPM AE 降至 0.474（ECMWF-IFS 0.507）。消融实验验证 ARP（ETS 提升 0.1–15.0%）、多模态数据（提升 5.6–13.0%）、NWP 引导（提升 2.0–9.3%）均有贡献。

## 局限

预测时效仅 12h；扩散模型推理较慢（1.253 s/sample）；依赖 ERA5-IFS 作为外生输入；TC 形成期和消散期性能较弱。
