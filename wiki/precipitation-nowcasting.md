---
title: "Precipitation Nowcasting"
type: concept
tags:
  - weather-forecasting
  - precipitation
  - nowcasting
  - radar
  - satellite
created: 2026-07-16
last_updated: 2026-08-30
source_count: 5
confidence: high
status: active
---

# Precipitation Nowcasting

降水临近预报（Precipitation Nowcasting）是指对 0–8 小时短时降水进行高分辨率预测的任务。与中期天气预报（1–15 天）不同，临近预报关注的是局部、快速演变的对流系统，对空间分辨率和时效性要求极高[^src-qcgs][^src-rainpro]。

## 为什么降水特别难

尽管基于 ERA5 的数据驱动全球预报模型（FourCastNet、Pangu-Weather、[[graphcast|GraphCast]]、ClimaX、FuXi、GenCast）已在中程超越 NWP[^src-graphcast]，降水仍然是突出难点。根本原因在于**尺度不匹配**：全球模型运行在数十公里粗分辨率上，而降水关键特征（局部暴雨、对流单体）出现在亚网格尺度、间歇且局部地涌现[^src-qcgs]。此外，中期模型使用的 ERA5 再分析数据存在表面变量和降水的系统性偏差[^src-rainpro]。

## 方法演进

### 传统方法
- **光流法**：Lucas-Kanade 等对雷达反射率场做外推（如 PySTEPS），预报技巧受限于雷达保真度，且假设恒定运动和强度，随预报时效增长迅速退化[^src-rainpro]
- **统计插值**：Barnes 插值、Kriging、最优插值从雨量计点观测构建网格化降水场，但模糊尖锐边界、对站点密度敏感

### 深度学习时代
- **ConvLSTM**（Shi et al., 2015）：开创性工作，将卷积嵌入 LSTM 做时空预报
- **U-Net 变体**：RainNet（Ayzel et al., 2020）、SmaAt-UNet（Trebing et al., 2021）、Broad-UNet（Fernández & Mehrkanoon, 2021）广泛用于雷达降水预报[^src-rainpro]
- **GAN-based**：DGMR（Ravuri et al., 2021, Nature），深度生成式雷达降水预报
- **Transformer**：Earthformer（Gao et al., 2024）用时空 Transformer 做地球系统预报[^src-rainpro]
- **扩散模型**：PreDiff（Gao et al., 2023）、DiffCast（Yu et al., 2024）、CasCast（Gong et al., 2024）在短时效实现强性能

### 多源数据融合 + 环境条件化

突破雷达限制的关键方向是融合多源异构数据。[[rainpro|RainPro-8]]（ICLR 2026）首次在欧洲以 2km/px、10min 间隔实现 8 小时概率降水预报，整合雷达（RainViewer）、卫星（EUMETSAT 11 通道）、NWP（GFS 122 变量）和地形数据，36.7M 参数 U-Net+MaxViT 架构，性能超越运营 NWP 系统 65%[^src-rainpro]。

其核心创新包括：
- **[[ordinal-consistent-loss|有序一致性损失]]**：通过条件概率公式强制降水强度类别单调性，替代传统交叉熵[^src-rainpro]
- **单次前向预测**：48 个预报时效编码到通道维度一次输出，推理快 48× 且保持时序一致性[^src-rainpro]
- **时效权重衰减**：指数衰减加权训练，兼顾短时效精度和长时效稳定性[^src-rainpro]

[[metnet|MetNet]] 系列（Google）在 8–24h 美国概率降水预报上达到 SOTA，但依赖 227M 参数和 512 TPU v3 的大规模训练，代码和数据未公开，且使用交叉熵损失忽略强度序数关系[^src-rainpro]。

[[storminsight|StormInsight]]（ICML 2026）将范式从 2D 外推推向**环境条件化的 3D 垂直动力学推理**——三分量编码（对流状态 + 垂直交互 + 大气环境）+ Conditional Flow Matching 分层调制。在美国和法国 [[stormbench|StormBench]] 上 MAE 降低 12.4%，mCSI 提升 34.0%，尤其在快速增强和消散阶段优势显著[^src-storminsight]。其核心洞见是：临近预报的最大局限不在于观测风暴，而在于错失**风暴如何在环境条件下演化**[^src-storminsight]。参见 [[environment-conditioned-nowcasting|环境条件化临近预报]]。

### 雷达困境
上述方法几乎全部假设雷达为主要输入。但雷达网络成本高、地理覆盖有限，主要适用于欧美等地区。雷达分辨率固定，无法表征亚尺度过程[^src-qcgs]。

## 无雷达路线

| 路线 | 代表方法 | 优缺点 |
|------|----------|--------|
| 纯卫星 | Sat2Radar (NPM, Park et al. 2025) | 全球覆盖，但偏差大、固定分辨率 |
| 卫星-站点融合 | QCGS (Kim et al., ICLR 2026) | 分辨率灵活、精度高，但依赖 AWS 密度；核心依赖 [[gaussian-splatting|Gaussian Splatting]] 和 [[implicit-neural-representation|INR]] |
| 多源融合 + 概率预报 | [[rainpro|RainPro-8]] (ICLR 2026) | 8h 欧洲全覆盖，多源数据（雷达/卫星/NWP/地形），概率输出，36.7M 参数高效架构[^src-rainpro] |
| 传统插值 | Kriging, Barnes | 无需训练，但模糊边界 |

## 与极端天气预测的关系

临近预报与 [[extreme-weather-forecasting|极端天气预测]] 高度相关：强降水本身就是高影响极端事件。但临近预报关注短时（<8h）局地降水场重建，极端天气预测关注更广泛的事件类型（洪涝、热浪、闪电等）和更长的预报时效。[[qcgs|QCGS]] 的降水场生成能力可作为极端降水预测的数据同化前端[^src-qcgs]。[[rainpro|RainPro]] 的概率输出为 8 小时内各强度等级提供校准的不确定性估计，可辅助极端降水预警决策[^src-rainpro]。

[^src-qcgs]: [[source-qcgs]]
[^src-rainpro]: [[source-rainpro]]
[^src-storminsight]: [[source-storminsight]]
[^src-graphcast]: [[source-graphcast]]

## 热带气旋降水预测

与常规降水临近预报不同，[[tropical-cyclone-precipitation-forecasting|热带气旋降水预测]]聚焦随 TC 中心移动的动态 10°×10° 窗口，而非固定地理区域[^src-tcp]。将常规方法直接迁移效果不佳——现有 DL 模型在 TC 降水任务上甚至不如 Persistence 基线[^src-tcp]。

[[tcp-diffusion|TCP-Diffusion]]（ICML 2025）是论文自称首个基于深度学习的全球 TC 降水预测工作，核心创新包括：
- **[[adjacent-residual-prediction|ARP]]**：将预测目标从绝对降水值改为相邻时间步的变化量，赋予模型变化感知能力
- **多模态编码器**：3D CNN 编码 2D 降水/环境数据 + MLP-Transformer 编码 TC 属性 + ResNet-18 编码 ERA5-IFS 预测
- **NWP 集成**：以低成本 ERA5-IFS 为条件引导扩散去噪，论文报告在 TIGGE 对比中超越 ECMWF-IFS（ETS-6 0.412 vs 0.302、TPM AE 0.474 vs 0.507）[^src-tcp]

在 12h 预测窗口内，ETS-24 达 0.147（PreDiff 仅 0.119），ETS-60 是唯一超越 Persistence 的 DL 模型[^src-tcp]。

[^src-tcp]: [[source-tcp]]
