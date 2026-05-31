---
title: "Source: UrbanFM — Scaling Urban Spatio-Temporal Foundation Models"
type: source-summary
tags:
  - spatio-temporal
  - foundation-model
  - traffic-forecasting
  - transformer
  - scaling-laws
  - zero-shot
  - imputation
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# Source: UrbanFM — Scaling Urban Spatio-Temporal Foundation Models

**Wei Chen, Yuqian Wu, Junle Chen, Xiaofang Zhou, Yuxuan Liang** (HKUST(GZ) / HKUST). arXiv:2602.20677v2 (Mar 2, 2026; initial submission Feb 9, 2026).

## 核心问题

城市系统的时空数据流编码了人类出行与城市演化的基本法则，但现有方法陷于"场景定制"困境——大多数模型针对单一区域/任务优化，缺乏通用建模能力[^src-urbanfm]。受 Sutton 的"苦涩教训"启发，UrbanFM 采用 scaling 为中心视角，系统研究城市时空基础模型的**扩展什么**和**如何扩展**两个核心问题[^src-urbanfm]。

## 三大挑战与解决方案

UrbanFM 从第一性原理分析出发，识别城市时空数据的三个关键科学属性并给出系统性 scaling 方案[^src-urbanfm]：

| 科学属性 | 对应地理定律 | Scaling 维度 | 解决方案 |
|---------|------------|-------------|---------|
| Heterogeneity（异质性） | Tobler's Second Law | Data Scaling | **WorldST**：100+ 城市、8 域、10 亿+数据点 |
| Correlation（相关性） | Tobler's First Law | Computation Scaling | **MiniST**：GD-Tree 贪婪聚类+时空 patching |
| Dynamics（动态性） | Complex System Theory | Architecture Scaling | **UrbanFM**：极简 self-attention，最小归纳偏置 |

### 1. WorldST 数据规模化

三阶段数据处理流水线[^src-urbanfm]：

- **多源采集**：开放政府平台（NYC Open Data、TfL）+ 学术仓库（UCTB、UTD-19）+ 领域专用 API（PeMS），覆盖 100+ 城市 × 8 个 domain（Traffic Speed、Flow、Crowd、Taxi、Bike、Cellular、Occupancy）
- **统一摄入**：格式自适应解析器 + 频率同步（统一到 5 分钟 ∆t 采样率）
- **质量控制**：静态死传感器剔除（方差≈0）、3σ 异常截断、线性插值预填补小间隔缺失

WorldST 在空间覆盖和时间跨度上远超 UniST、OpenCity、BigCity 达 33–145 倍[^src-urbanfm]。

### 2. MiniST 计算规模化

将连续时空场离散化为统一可学习计算单元（类比 NLP 的 token）[^src-urbanfm]：

- **贪心容量约束聚类**：基于 KD-Tree 按地理坐标将 N 个空间节点分组为大小 Sp 的固定簇，不足时 binary masking
- **时空 patching**：定义 patch X(k,t) ∈ R^(Sp×Tp)，将不规则空间结构转化为规则 patch 样本
- **优势**：无需邻接矩阵，token 结构独立可大规模并行训练；本地聚合显式编码"近者更相关"的地学原理[^src-urbanfm]

### 3. UrbanFM 架构规模化

极简 Transformer 设计，最小归纳偏置，完全依赖 self-attention 从海量数据自主学习动态依赖[^src-urbanfm]：

- **分解时空注意力**：先 temporal attention（空间节点内建模时间依赖），再 spatial attention（时间步间建模空间依赖）
- **ST-RoPE 位置编码**：T-RoPE 编码相对时间距离（支持变长上下文外推）；S-RoPE 编码线性化样本内的相对空间序（无需邻接矩阵）
- **RevIN + 生成式建模目标**：实例归一化处理非平稳性；未来时间步用零噪声 mask，强制模型从噪声上下文中重建未来信号，统一了 forecasting 和 imputation[^src-urbanfm]

## 实验评估

### EvalST 基准

构建迄今为止最大规模的城市场景时空评估基准 EvalST，12 个数据集，覆盖 4 国 7 城、传感器和网格两种格式、流量/速度/占有率/路网/自行车/出租车/轨迹等多个领域，时间跨度超过 10 年[^src-urbanfm]。对比 22 个 baseline（包括专家模型 STAEformer/STID/D2STGNN/GWNET + 时序基础模型 TimesFM++/Moirai/Time-MoE/Chronos + 时空基础模型 FactoST/OpenCity）。

### 关键结果

- **零样本泛化**：在所有场景下一致最优，MAPE 优于现有时空基础模型 39.0%–70.2%；零样本性能超越甚至媲美专职专家模型的 full-shot 训练性能。长时传感器预测中 UrbanFM (17.0 MAPE) 超越 D2STGNN (20.5) 和 STID (19.9)[^src-urbanfm]
- **小样本微调**：仅 10% 目标域样本微调后进一步提升 28.2%–65.2% 超越 full-shot 专家模型[^src-urbanfm]
- **跨任务泛化**：预训练阶段无任何 imputation 目标的情况下，在 PEMS 四个数据集上的点和块缺失填补均取得最优，显著超越 MICE/SVD 等经典方法[^src-urbanfm]
- **Scaling 特性**：模型深度增加（2→8 层）持续降低误差；预训练数据量指数扩展下 MAPE 呈幂律衰减，未观察到性能饱和[^src-urbanfm]
- **鲁棒性**：30% 零掩码+30% 高斯噪声注入下保持最高保真度，OpenCity 次之，专职专家模型（STAEformer）显著退化。归因于生成式预训练将信号恢复作为核心目标[^src-urbanfm]
- **效率**：A100 GPU 上推理时间约 Chronos 的 1/4、TimeMoE 的 1/10，同时 MAPE 最优（15.2%）[^src-urbanfm]

## 局限性

讨论于 Appendix E：隐私与匿名性、建模公平性、参数规模扩展受工程限制、多模态对齐（LLM）和决策优化是未来方向[^src-urbanfm]。

[^src-urbanfm]: [[source-urbanfm]]
