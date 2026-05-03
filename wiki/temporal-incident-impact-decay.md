---
title: "Temporal Incident Impact Decay (TIID)"
type: technique
tags:
  - traffic-forecasting
  - temporal-modeling
  - decay-modeling
  - incident-aware
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Temporal Incident Impact Decay (TIID)

TIID 是 [[igstgnn|IGSTGNN]] 的核心模块之一，负责显式建模事件影响随时间的动态消散过程。核心思想：事件影响不是单一静态脉冲，而是随时间衰减的动态过程，无法建模此衰减是长期预测误差的主要原因[^src-incident-guided-st-forecasting]。

## 问题动机

传统交通预测模型将事件影响作为静态特征注入，忽略其时间演化：
- 事件发生后，影响逐渐消散——网络逐步恢复
- 长期预测中，未建模衰减会导致持续高估事件影响
- 短期预测中，影响最强；长期预测中，影响趋近于零

## 机制

### 1. 初始事件上下文计算

从时空建模模块输出的基础预测 H_pred 出发，计算事件在发生时刻 t 的初始空间影响 C_init：

- 事件 Key 表示 K → 空间掩码（D）→ 局部化事件表示 K_context
- K_context ∥ 传感器特征 S ∥ 空间关系 D → MLP g_c → C_init

这确保事件只影响其空间连接的节点[^src-incident-guided-st-forecasting]。

### 2. 高斯衰减因子

对每个未来预测步 τ ∈ {1, ..., T_p}，使用高斯函数计算衰减：

ω_τ = exp(-τ² / 2σ²_t)

其中 σ_t 是控制衰减速率的超参数（IGSTGNN 中设为 1.0）。衰减因子向量 ω ∈ ℝ^{T_p}[^src-incident-guided-st-forecasting]。

### 3. 时间事件影响

C_temp = ω ⊗ (C_init · W_c)

将初始影响通过线性变换后，与衰减因子进行广播乘法，得到时间调制后的事件影响张量[^src-incident-guided-st-forecasting]。

### 4. 叠加到基础预测

Ŷ = g_out(H_pred + C_temp)

将事件影响叠加到时空骨干网络的基础预测上，通过 MLP 预测头生成最终预测[^src-incident-guided-st-forecasting]。

## 高斯衰减的物理合理性

高斯衰减的选择基于事件影响的物理特性：
- 事件发生时影响最强（τ=0 时 ω=1）
- 随时间距离增加，影响平滑衰减
- 长期预测中事件影响趋于零——符合网络恢复的现实

相比指数衰减，高斯衰减在短期更强、长期更快趋零，与交通恢复的典型模式一致[^src-incident-guided-st-forecasting]。

## 优势验证

- 移除 TIID 模块后，Alameda 上 MAE 从 12.69 升至 13.23（退化 4.2%），三个数据集一致退化[^src-incident-guided-st-forecasting]。
- 长期预测（Horizon 12）优势最显著：Orange 上 IGSTGNN 仍保持 4.1% MAE 优势，直接验证 TIID 捕获交通恢复动态的有效性[^src-incident-guided-st-forecasting]。
- ICSF + TIID 组合时协同效应最强（D2STGNN + 双模块: MAE 提升 13.23% > 单独提升之和）[^src-incident-guided-st-forecasting]。

## 即插即用性

TIID 可独立集成到其他 STGNN 骨干网络。在 D2STGNN 上单独集成 TIID 时 MAE 提升 9.85%（Alameda），与 ICSF 形成功能互补[^src-incident-guided-st-forecasting]。

## 相关技术

- [[incident-context-spatial-fusion|ICSF]] — 配套的空间融合模块
- [[guided-layer-normalization|GLN]] — ConFormer 的条件机制（不同路线）
- [[accident-aware-traffic-forecasting]] — 事件感知交通预测问题域

[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
