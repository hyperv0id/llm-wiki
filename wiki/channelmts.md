---
title: "ChannelMTS"
type: entity
tags:
  - channel-prediction
  - multimodal-time-series
  - kdd-2026
created: 2026-04-30
last_updated: 2026-05-03
source_count: 0
confidence: medium
status: active
---

# ChannelMTS

**ChannelMTS** (Channel Multi-modal Time-Series) 是香港科技大学（广州）赵海虹、郑子楠、自动化和李佳教授团队提出的高铁多模态信道预测框架，发表于 KDD 2026[^source]。

## 核心问题

高铁通信中，列车高速移动（300 km/h）导致环境快速变化（平原→农村→城市→隧道），信道状态随之剧烈变化。传统单模态方法仅利用历史信道数据，难以捕捉环境动态，导致预测性能不佳。

## 核心创新

1. **环境快照表示**：$E_t = \{p_t, K_t, \zeta_t\}$（位置、K因子、RMS延迟）
2. **检索增强统计信道 (RAGC)**：从预缓存高铁地图中检索相似环境对应的统计信道
3. **模态对齐与自适应融合**：中位数+IQR 分布对齐 + 动态权重融合
4. **未来环境信息利用**：利用铁���轨迹预定义特性，有效利用未来环境信息

## 技术架构

### 三组件设计

1. **Channel Predictor**：时间序列骨干（Transformer/RNN/CNN/Linear）从历史信道预测未来
2. **Environmental Dynamics Encoder**：环境快照 → RAGC 检索 → Transformer 编码 → 初始信道映射
3. **Modality Alignment & Fusing**：分布对齐 + 自适应融合

### 分布对齐公式

$$\hat{E} = \frac{E - \mu_E}{\sigma_E} \times \sigma_C + \mu_C$$

其中 $\mu$ 为中位数，$\sigma$ 为四分位距（IQR）。

## 实验结果

### 离线实验

| 数据集 | ChannelMTS | 最佳基线 | 提升 |
|--------|-----------|---------|------|
| HSR I MSE | **0.0722** | 0.0859 | 16% |
| HSR II MSE | **0.0767** | 0.1139 | 33% |
| VSR I MSE | **0.1675** | 0.2569 | 35% |

### 线上 A/B 测试

在真实 5G NR MIMO 系统上：
- MSE 降低 **82%-92%**
- COS 降低 **71%-86%**

### 部署效果

- 频谱效率：4 → 8.75 bps/Hz
- 下行峰值速率：80 → >175 Mbps
- 隧道场景：从 0 bps/Hz 提升到可支持稳定通话

## 相关页面

- [[source-channelmts]] — 论文详细摘要
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测概念
- [[mindts]] — MindTS 多模态异常检测（同为多模态 TS，不同任务）
- [[multimodal-time-series-anomaly-detection]] — 多模态异常检测任务

---

## 引用

[^source]: https://doi.org/10.1145/3770854.3783957 — KDD 2026