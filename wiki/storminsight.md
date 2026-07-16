---
title: "StormInsight"
type: entity
tags:
  - weather-forecasting
  - nowcasting
  - precipitation
  - flow-matching
  - convection
  - multi-modal
  - icml-2026
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# StormInsight

StormInsight 是由 HKUST-GZ 和广州气象台提出的气象学启发的对流临近预报框架，发表于 ICML 2026[^src-storminsight]。它将传统 2D 雷达回波外推重新定义为**环境条件化的三维垂直结构动力学推理**。

## 问题动机

2025 年德州 Kerrville 五百年一遇暴雨——250mm 降雨在 3 小时内 ——暴雨虽被探测到，但向极端强度的突然转变未被及时识别，预警滞后，山洪夺去 130 余人生命。这暴露了现有临近预报的根本局限：最大风险不在于观测风暴，而在于**错失风暴如何演化**[^src-storminsight]。

## 两大核心挑战

1. **异步跨层交互（C1）**：对流反馈在不同垂直层之间以错位、状态依赖的时间尺度传播——近地面水汽辐合可能比中层潜热释放或云顶冷却早几十分钟触发，也可能根本不触发。这一时序异步性打破了多层层叠模型的隐含同步假设[^src-storminsight]。

2. **环境依赖敏感性（C2）**：局地前兆的影响并非内禀——相同低层辐合模式在有利环境中可触发深对流，在中层干侵入下可被完全抑制。模型需动态地用大尺度环境强迫来门控局地风暴演化[^src-storminsight]。

## 架构

### Storm Evolution Encoder

三分量结构化编码[^src-storminsight]：

- **Convective State Encoder**：通过 SetConv 统一将卫星（高）、雷达（中）、站点（低）投影到共享经纬度网格，MSIM 多尺度特征提取，VAE 编码雷达潜变量，FiLM 融合卫星/站点条件。
- **Vertical Interaction Encoder**：MoE 驱动 4 条定向跨层路径——$E_{M \to H}$（中层潜热→高层加速输送）、$E_{H \to M}$（云砧辐射冷却→调制中层上升）、$E_{M \to L}$（对流下沉→扰乱边界层）、$E_{L \to M}$（边界层加热→触发新单体）。每条专家用 Transformer 骨干 + Gate-of-Time 模块处理异步时序。
- **Atmospheric Environment Encoder**：Multi-mesh Message Passing（GraphCast 风格）在多分辨率球面网格上编码 ERA5 的压力层 + 单层变量（65+ 变量），保留高阶垂直依赖和天气尺度动力结构。

### Convective System Decoder

基于 Conditional Flow Matching 的生成式传播[^src-storminsight]：

- **Global Adapter**：将环境编码 $Z_{env}$ 通过 AdaLN 注入流匹配骨干的起止端——$v_{\theta}(z_t, t, Z_{cond}) = v_{core} + F_{\theta}(Z_{env})$，同构于大气方程 $\partial S / \partial t = D(S) + F_{env}$。
- **Local Adapter**：垂直交互特征 $Z_{expert}$ 通过 AdaLN 注入核心网络每一注意力层，调控向量场几何。

## 性能

在 [[stormbench|StormBench]]（美法双区域，多源观测+ERA5 再分析）上，StormInsight 全面超越 ConvLSTM、SimVP、Earthformer、NowcastNet、AlphaPre 等 10 个基线[^src-storminsight]：

- MAE 降低 **12.4%**
- mCSI 提升 **34.0%**
- 在强对流核心（CSI₂₁₉）和长时效（120min）上优势尤为显著
- 推理延迟约 380ms/25 帧（单 H100），满足业务部署约束

[^src-storminsight]: [[source-storminsight]]
