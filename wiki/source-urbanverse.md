---
title: "Source: UrbanVerse — Learning Urban Region Representation Across Cities and Tasks"
type: source-summary
tags:
  - urban-computing
  - region-representation
  - foundation-model
  - cross-city
  - diffusion-model
  - transformer
  - random-walk
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# Source: UrbanVerse — Learning Urban Region Representation Across Cities and Tasks

**Fengze Sun, Egemen Tanin, Shanika Karunasekera, Zuqing Li, Flora D. Salim, Jianzhong Qi** (University of Melbourne / UNSW). arXiv:2602.15750 (Feb 17, 2026).

## 核心问题

城市区域表征学习（urban region representation learning）长期陷于"城市中心主义"困境：现有方法将整个城市建模为图，区域作为图上节点，通过重建、对比或自监督学习区域 embedding[^src-urbanverse]。这些 city-centric 方法导致学到的 embedding 与城市全局结构深度耦合，在跨城市时彻底失效[^src-urbanverse]。同时，现有方法为每个下游任务独立训练预测器，无法利用跨任务共享的底层城市规律[^src-urbanverse]。

UrbanVerse 的目标是将城市区域表征学习从 city- and task-specific 范式提升为 foundation-style 模型，实现跨城市、跨任务的泛化[^src-urbanverse]。

## 核心贡献

### 1. 方法论翻转：从城市中心到区域中心

现有方法站在城市角度看区域，UrbanVerse 站在区域角度看待自身[^src-urbanverse]。关键操作：将城市切割为 150m 边长的正六边形网格 cell，丢弃城市全局图结构，仅依赖局部特征（POI 分布 + 邻域结构）进行跨城市可迁移的表征学习[^src-urbanverse]。

### 2. CELearning：跨城市 Cell Embedding 学习

以网格 cell 为基本单元，使用 Node2vec 的 p/q 参数化随机游走生成 cell 序列，通过 Encoder-Decoder Transformer 以 mask-reconstruct（30% mask 率，MSE 重建被 mask 的 POI 特征）学习 cell embedding[^src-urbanverse]。随机游走的随机性提供隐式数据增强，增强泛化性和鲁棒性[^src-urbanverse]。最终通过继承自 FlexiReg 的 AdaRegionGen 模块，将该区域内所有 cell embedding 按重叠面积加权求和得到区域 embedding[^src-urbanverse]。

关键设计选择：仅使用 15 维 POI 特征 + 邻居 ID，不使用卫星图、街景、LLM 文本——最小特征集策略确保了跨城市时特征空间对齐[^src-urbanverse]。

### 3. HCondDiffCT：异质条件扩散回归

将城市预测问题从点估计提升为条件分布估计 p(y|h, u)[^src-urbanverse]：
- **RegCondP（区域条件先验引导）**：从训练集中按 embedding 余弦相似度检索 Top-5 相似区域的真实标签值作为先验 ỹ，注入扩散过程——前向过程从 y₀ 向 ỹ 插值而非向高斯噪声[^src-urbanverse]
- **TaskCondD（任务条件去噪器）**：扩散时间步 t 和任务指示符 u 通过 learnable embedding 编码后，以元素级调制（element-wise modulation）控制去噪网络的行为[^src-urbanverse]

HCondDiffCT 是即插即用的通用模块，可集成到任何现有城市表征学习模型中增强下游任务效果[^src-urbanverse]。

## 实验证据

3 个城市 (NYC/CHI/SF) × 6 个任务 (Crime / Check-in / Service Call / Population / Carbon / Nightlight)，7 个 baseline (HREP / RegionDCL / UrbanCLIP / CityFM / GeoHG / GURPP / FlexiReg)[^src-urbanverse]：

- **Cross-city（18 测试设定）**：UrbanVerse 一致超越所有 baseline。Crime R²=0.724 vs FlexiReg 0.545（NYC target，+32.9%）；Crime R²=0.814 vs FlexiReg 0.599（SF target，+35.9%，全表最大提升）；Population 总体提升 10.5-39.4%[^src-urbanverse]
- **郊区泛化（Staten Island）**：Population R²=0.781 vs FlexiReg 0.609（+28.2%）；Carbon R²=0.945 vs FlexiReg 0.869（+8.7%）[^src-urbanverse]
- **HCondDiffCT 外挂能力**：集成到 GURPP / UrbanCLIP / HREP / HAFusion 后，所有 24 设定一致提升。GURPP-DiffCT nightlight R² 从 0.035 → 0.171（+388.6%）；UrbanCLIP-DiffCT carbon R² 从 0.021 → 0.204（+871.4%）[^src-urbanverse]
- **Same-city**：FlexiReg 在同城市设置下与 UrbanVerse 持平甚至略优（符合预期），但 UrbanVerse 仅使用 POI+邻居特征，特征集更小、泛化性更强[^src-urbanverse]
- **新任务适应**：fine-tuning 与从头训练 R² 差距 < 0.02[^src-urbanverse]

## 局限

- 仅使用 POI + 空间邻接特征，无法利用多模态数据[^src-urbanverse]
- 无时间维度——纯粹静态 snapshot，不适用于交通预测等时序任务[^src-urbanverse]
- 必须 >1 个城市训练，无法 true one-shot zero-shot（单城市时退化为 FlexiReg）[^src-urbanverse]
- 仅覆盖连续值回归任务，未展示分类任务表现[^src-urbanverse]
- 训练城市均为美国大城市，全球南方城市系统性评估不足[^src-urbanverse]

## 与相关工作的关系

UrbanVerse 与 [[urbanfm|UrbanFM]]（arXiv 2026 同期）构成城市基础模型的两个互补方向：UrbanFM 处理交通流/速度等时空序列预测，UrbanVerse 处理犯罪/人口/碳排放等区域属性预测[^src-urbanverse]。UrbanVerse 继承 [[flexireg|FlexiReg]]（同组 2025 KDD）的 AdaRegionGen 模块，但将 cell 表征学习从城市级 mask-reconstruct 升级为跨城市随机游走范式[^src-urbanverse]。

[^src-urbanverse]: [[source-urbanverse]]
