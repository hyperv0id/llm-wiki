---
title: "StormInsight: Hierarchical Environmental Forcing and Vertical Coupling for Convective Systems Evolution"
type: source-summary
tags:
  - weather-forecasting
  - nowcasting
  - precipitation
  - flow-matching
  - convection
  - multi-modal
  - reanalysis
created: 2026-07-23
last_updated: 2026-07-23
source_count: 0
confidence: low
status: active
---

# StormInsight: Hierarchical Environmental Forcing and Vertical Coupling for Convective Systems Evolution

**Authors:** Jun Chen, Yan Fang, Minghui Qiu, Yueran Qiu, Lin Chen, Shuxin Zhong (HKUST-GZ, Guangzhou Meteorological Observatory), Yu Zhang, Kaishun Wu
**Venue:** ICML 2026

## 核心贡献

StormInsight 是一个气象学启发的临近预报框架，将对流系统演化的建模从 2D 雷达回波外推重新定义为**环境条件化的 3D 垂直结构动力学推理**。论文引入了两个核心组件：

1. **Storm Evolution Encoder**：三分量编码方案——Convective State Encoder 提取多模态观测（卫星/雷达/站点）的逐层物理状态；Vertical Interaction Encoder 用 MoE 建模跨层定向热力学耦合（如低层辐合→中层潜热释放→高层云砧扩展）；Atmospheric Environment Encoder 通过 Multi-mesh Message Passing 编码 ERA5 再分析的天气尺度环境约束。

2. **Convective System Decoder**：基于 Conditional Flow Matching 的环境感知时序传播——Global Adapter 注入大尺度环境强迫作为 AdaLN 全局条件，Local Adapter 将垂直跨层反馈注入每一注意力层的 scaling/shifting，实现「相同局地前兆→不同环境条件下产生不同对流结果」。

## 关键结果

在自建的 StormBench 基准（整合美国 SEVIR 事件数据和法国 MeteoNet 长期序列，含多源观测 + ERA5 再分析的 65+ 变量）上，StormInsight 始终超越基线模型（ConvLSTM、SimVP、Earthformer、NowcastNet、AlphaPre 等），MAE 降低 12.4%，mCSI 提升 34.0%。消融实验证实：移除 Atmospheric Environment Encoder 导致强对流（CSI₂₁₉）显著退化；移除 Vertical Interaction Encoder 则削弱对流初生前兆的捕获；移除任一定向专家均导致性能下降，证明跨层交互是协同涌现而非单一主导路径。

## 局限与前景

当前框架主要作用于多层 2D 表示，未充分利用雷达体的完整 3D 扫描；偏振雷达变量（ZDR、KDP）尚未纳入。论文还展示了结合 CMA 实时业务分析替换 ERA5 的初步部署验证，提示了走向全业务化部署的可行路径。


