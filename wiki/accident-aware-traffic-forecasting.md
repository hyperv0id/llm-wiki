---
title: "Accident-Aware Traffic Forecasting"
type: concept
tags:
  - traffic-forecasting
  - anomaly-detection
  - resilience
  - incident-aware
created: 2026-04-28
last_updated: 2026-05-03
source_count: 2
confidence: high
status: active
---

# Accident-Aware Traffic Forecasting

**Accident-aware (incident-aware) traffic forecasting** is an emerging paradigm that explicitly models the disruptive impact of external events on prediction systems. Traditional traffic forecasting models assume relatively stationary patterns and excel at capturing recurring dynamics (e.g., rush hour congestion), but they falter when accidents or other non-recurrent incidents create non-stationary perturbations with distinctive directional shockwaves through transportation networks[^src-conformer][^src-incident-guided-st-forecasting].

> [!note] 术语扩展
> 最初的研究聚焦于交通事故（accident-aware），但更广泛的非重复性事件（恶劣天气、道路施工、车辆故障等）同样产生显著影响。IGSTGNN (KDD 2026) 将范围扩展为"incident-guided"，覆盖 6 类事件[^src-incident-guided-st-forecasting]。

## Problem Statement

Traffic accidents introduce several challenges that standard models cannot handle:

1. **Non-stationary perturbations** — events create sudden speed drops and complex propagation patterns that deviate from normal traffic dynamics[^src-conformer][^src-incident-guided-st-forecasting].

2. **Directional shockwaves** — disruptions spread asymmetrically through connected road networks in nonlinear ways; upstream impacts are far more severe than downstream[^src-conformer][^src-incident-guided-st-forecasting].

3. **Data limitations** — most traffic datasets lack detailed incident information, offering only basic flow data[^src-conformer].

4. **Dynamic temporal evolution** — an event's influence is not a static impulse but decays over time as the network recovers; failure to model this decay causes severe long-term prediction errors[^src-incident-guided-st-forecasting].

5. **Heterogeneous spatial influence** — even two events between the same sensor pair may produce different impacts depending on distance and direction[^src-incident-guided-st-forecasting].

Research shows that accidents can increase travel times by 37-43% compared to normal conditions[^src-conformer].

## Approaches

### ConFormer (KDD 2026)

The first Transformer-based model capable of operating on very large graphs while explicitly modeling accident propagation:

- **Accident-aware graph propagation** — models how disruptions spread through traffic networks using K-hop Laplacian operations[^src-conformer].
- **Guided Layer Normalization (GLN)** — dynamically adjusts normalization parameters based on traffic conditions[^src-conformer].
- **Conditional self-attention** — incorporates contextual condition representations into attention computation[^src-conformer].

### IGSTGNN (KDD 2026)

[[igstgnn|IGSTGNN]] 将事件范围从交通事故扩展到广义非重复性事件（6 类：Hazard、Accident、Breakdown、Weather、Other、Police），通过两个即插即用模块显式建模事件影响：

- **[[incident-context-spatial-fusion|ICSF]]** — 注意力融合事件特征 + 传感器属性 + 空间关系张量 D，为每个节点生成定制化的事件感知初始状态[^src-incident-guided-st-forecasting]。
- **[[temporal-incident-impact-decay|TIID]]** — 高斯衰减函数 ω_τ = exp(-τ²/2σ²) 显式建模事件影响的时间消散，叠加到基础预测上[^src-incident-guided-st-forecasting]。

三个数据集上全面超越 SOTA（Alameda 平均 MAE 比次优低 5.65%），ICSF + TIID 可即插即用集成到多种骨干网络（AGCRN、GWNET、STTN、D2STGNN）[^src-incident-guided-st-forecasting]。

### ConFormer vs IGSTGNN

| 维度 | ConFormer | IGSTGNN |
|------|-----------|---------|
| 事件范围 | 交通事故 | 广义非重复性事件 |
| 空间建模 | K-hop 拉普拉斯图传播 | 注意力 + 空间关系张量 D |
| 时间建模 | 无显式衰减 | 高斯衰减函数 |
| 即插即用 | ✗ | ✓ |

### Key Results

ConFormer achieves up to **10.7% improvement** in accident scenarios compared to STAEFormer[^src-conformer]. IGSTGNN achieves **5.65% average MAE improvement** over second-best on Alameda, and ICSF+TIID integration improves D2STGNN by **13.23%**[^src-incident-guided-st-forecasting].

## Datasets

Two enriched benchmark datasets with accident annotations:

- **Tokyo**: 1,843 highway segments, accident + regulation data from JARTIC
- **California**: Bay Area (2,352 sensors) + San Diego (716 sensors), accident data from US Accidents database[^src-conformer]
- **XTraffic-based (Alameda / Contra Costa / Orange)**: 521-990 nodes, 5,587-18,700 incidents, California 2023, 5-min granularity with time-aligned incident records[^src-incident-guided-st-forecasting]

## Related Concepts

- [[traffic-forecasting]] — general traffic prediction
- [[guided-layer-normalization]] — conditional normalization technique (ConFormer)
- [[incident-context-spatial-fusion]] — ICSF spatial fusion technique (IGSTGNN)
- [[temporal-incident-impact-decay]] — TIID temporal decay technique (IGSTGNN)
- [[conformer]] — the primary model for accident-specific forecasting
- [[igstgnn]] — the primary model for broader incident-guided forecasting

[^src-conformer]: [[source-conformer]]
[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]