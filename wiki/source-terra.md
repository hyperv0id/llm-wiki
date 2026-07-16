---
title: "Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth"
type: source-summary
tags:
  - dataset
  - spatiotemporal
  - multimodal
  - benchmark
  - 2024
created: 2026-07-07
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth

> Wei Chen, Xixuan Hao, Yuankai Wu, Yuxuan Liang (CityMind-Lab, HKUST(GZ) / Sichuan University). NeurIPS 2024 Track on Datasets and Benchmarks.

**Terra** 是一个大规模多模态地球时空数据集，旨在突破现有数据集在空间覆盖范围、时间跨度和模态多样性方面的局限，为**时空通用智能**研究提供基础平台。[^src-terra]

---

## 规模与覆盖

Terra 将全球划分为 0.1°×0.1° 的栅格网格，覆盖 **6,480,000 个网格区域**，提供 **1979–2024 年（45 年）** 每 3 小时的完整气象观测，总计超过 **6.82×10¹² 条**数值记录 [^src-terra]。数据支持 3 种空间分辨率（0.1° / 0.5° / 1°）和 3 种时间粒度（3 小时 / 日 / 月），共 9 种变体。

### 时间序列模态

基于 GloH2O 项目的 MSWX 和 MSWEP 产品，包含 10 个气象变量：[^src-terra]
- 降水（mm/3h）、气温（°C）、地表气压（Pa）、风速（m/s）
- 相对/绝对湿度（%、g/g）、短波/长波辐射（W/m²）
- 日最低/最高温（因分辨率受限而排除）
- MSWEP 降水估计替代 MSWX 以利用其多源融合优势（雨量计 + 卫星 + 再分析）

### 文本模态

为每个栅格区域提供两级地理文本信息：[^src-terra]
- **元数据**：柯本气候分类、平均海拔（ETOPO2v2）、土地覆盖（C3S 全球 38 类）、所属国家
- **LLM 生成描述**：使用 LLaMA3 结合空间提示工程（Spatial Prompt Engineering），注入元数据事实以减少幻觉，生成涵盖地理位置、气候、海洋影响、地形、季风、气流、植被和人类活动的段落级描述

### 图像模态

使用 GMT/PyGMT 和 ArcGIS 为每个网格渲染多类地理图像：[^src-terra]
- Earth Geoid、Free-Air Anomaly Errors、Magnetic Anomaly
- Earth Mask（水陆分布）、Relief（地形起伏）、Vertical Gravity Gradient
- 卫星遥感图像

---

## 关键特性

与现有数据集（GeoLife、SEVIR、ClimSim、Digital Typhoon 等）相比，Terra 是唯一同时满足以下条件的：[^src-terra]
- **大规模**：6,480,000 网格 × 45 年
- **细粒度**：0.1° 空间 × 3 小时时间
- **多模态**：时间序列 + 文本 + 图像

---

## 实验验证

论文在两个任务上验证了 Terra 的实用性：[^src-terra]

1. **时空预测（降水预报）**：在 7/15/30 天预测场景下评估 10 种基线。TimesNet 和 STID 表现最优。有趣的是，时空图模型（GWNet、STGCN）因空间内存消耗过大在全局预测中失效——揭示了高效时空模型的迫切需求。

2. **空间变量预测**：使用 SatCLIP/GeoCLIP/CSP 进行位置编码预测，以及 CLIP/UrbanCLIP/UrbanVLP 进行视觉-语言预测。SatCLIP 在降水/风速/温度预测中全面领先，UrbanVLP 在文本-图像融合任务中表现最佳。

---

## 与 [[source-exost]]、[[source-aurora]] 的关系

- **ExoST**（Select, then Balance，arXiv 预印本 2509.05779，未经同行评审截至 2026-07）提出时空预测外生变量的 select-then-balance 建模思路；Terra 可为其提供全球尺度的多模态外生数据基础。
- **Aurora**（通用生成式多模态时间序列预测基础模型）依赖大规模多模态数据集进行预训练；Terra 的全球覆盖和 multimodal 特性使其成为 Aurora 类模型的理想训练/评测平台。

---

## 局限性

- 图像/文本模态暂不支持 0.1° 空间分辨率（成本过高）。[^src-terra]
- LLM 生成文本存在知识时效性和幻觉风险，虽然空间提示工程有所缓解但并未根除。[^src-terra]
- 卫星遥感图像存在过时和不稳定的分发限制。[^src-terra]

---

## 引用

[^src-terra]: [[source-terra]]