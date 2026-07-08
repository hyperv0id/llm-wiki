---
title: "Spatio-Temporal Foundation Models: Vision, Challenges, and Opportunities"
type: source-summary
tags:
  - survey
  - foundation-model
  - spatiotemporal
  - multimodal
  - 2025
created: 2026-07-07
last_updated: 2026-07-08
source_count: 1
confidence: medium
status: active
---

# ST Foundation Models: Vision, Challenges, and Opportunities

> Bryan Hooi, Adam Goodge, Wee Siong Ng, See-Kiong Ng (Institute for Infocomm Research, A*STAR Singapore; National University of Singapore). arXiv:2501.09045v2, Feb 2025.

本文为时空基础模型（Spatio-Temporal Foundation Models, STFMs）提出了系统性的愿景，识别了关键聚合能力、现有研究的碎片化问题以及未来的发展机遇。[^src-st-foundation-models-survey]

---

## 核心贡献

- 提出 STFMs 的正式定义和四种泛化能力框架。[^src-st-foundation-models-survey]
- 对 6 个代表性 STFMs（UniST、OpenCity、UrbanGPT、ClimaX、Pangu-Weather、W-MAE）进行定性评估。[^src-st-foundation-models-survey]
- 识别 STFM 领域的关键挑战（碎片化、泛化差距、基准缺失），并展望未来方向。[^src-st-foundation-models-survey]

## 四种泛化能力框架

STFMs 应在以下四个维度实现泛化：[^src-st-foundation-models-survey]

1. **领域泛化（Domain Generalization）**：跨不同数据源和应用场景——例如交通流和交通事故（同一物理系统下强相关），或交通与疾病爆发（弱相关，存在负迁移风险）。
2. **空间泛化（Spatial Generalization）**：跨不同地理区域——例如从北京/纽约训练的交通模型泛化到其他城市。现有数据集严重偏向少数大城市（北京、纽约、伦敦），空间偏差风险显著。
3. **时间泛化（Temporal Generalization）**：跨不同时间段——不仅包括日/周末的自然变化，还包括渐进变化（如城市人口增长）和突发变化（如新景点开放或自然灾害）。
4. **尺度泛化（Scale Generalization）**：跨不同空间分辨率和时间频率——如全球天气模式（粗粒度）与区域微气候（细粒度）之间的迁移。

## 现有 STFMs 评估

论文对 6 个 STFMs 从上述四维度进行定性评估：[^src-st-foundation-models-survey]

**交通领域**（UniST、OpenCity、UrbanGPT）：
- UniST 和 OpenCity 在 21 个数据集上训练/评测，覆盖交通速度/流量、自行车/出租车需求等，评测了 ID 和 OOD 泛化。
- UrbanGPT 仅使用 4 个数据集（NYC 出租车/自行车/犯罪 + Chicago 出租车），OOD 测试有限。
- 空间覆盖严重受限——几乎全部来自美国和中国的少数城市。

**天气领域**（ClimaX、Pangu-Weather、W-MAE）：
- 均基于 CMIP6 和 ERA5 两大全球天气数据集训练，空间覆盖全球。
- 但仅评估少量变量（4 个），分辨率粗（ClimaX 最低 5.625°），难以泛化到细粒度区域预测。
- Pangu-Weather 为不同预测周期训练 4 个独立模型，与基础模型的统一范式相悖。

**关键发现**：现有 STFMs 高度碎片化为交通和天气两大阵营，彼此无交叉；空间覆盖和数据多样性严重不足；缺乏统一的基准和评估协议。[^src-st-foundation-models-survey]

## 未来方向

### 统一架构（Unified Architectures）
现有 STFMs 主要处理结构化网格数据，对其他 ST 数据类型（点参考、轨迹、事件）支持不足。需要在架构层面原生支持异构 ST 数据，并探索稀疏注意力、层次建模、混合架构（CNN+RNN+GNN）来克服 Transformer 的二次复杂度瓶颈。[^src-st-foundation-models-survey]

### 跨域协同（Cross-domain Synergies）
不同领域之间存在未被充分利用的共享模式——例如人类流动性模式与传染病传播密切相关。将天气、交通、社交媒体、流行病数据联合训练有望学到更鲁棒的 ST 表示。需注意领域间关系通常是方向性的（如天气影响交通但反之不成立），需引入因果学习来发现和利用这些关系。[^src-st-foundation-models-survey]

### 多模态训练（Multi-modal Training）
物理世界的信息本质上是多模态的——数值时间序列、遥感影像、文本描述、音频信号等。例如城市交通预测可融合摄像头画面、GPS 数据和社交媒体报告；作物产量预测可融合卫星影像、气象时间序列和土壤传感器数据。然而多模态 ST 数据对齐（时空分辨率差异巨大）仍是关键挑战。Terra 等数据集提供了初步探索。[^src-st-foundation-models-survey]

### 分布偏移适应（Adaptation to Distribution Shift）
ST 数据固有的动态性导致持续分布偏移。域对抗训练（domain-adversarial training）、元学习（meta-learning）和测试时自适应（test-time adaptation）是有前景的适应策略。[^src-st-foundation-models-survey]

## 交叉链接

- [[source-climax]] — ClimaX，本文评估的天气基础模型
- [[source-unist]] — UniST，本文评估的交通基础模型（prompt-empowered 统一预测框架）
- [[source-urbangpt]] — UrbanGPT，本文评估的 LLM 驱动的交通基础模型
- [[source-terra]] — Terra 多模态数据集，本文引用的多模态 ST 数据实例
- [[source-stfm-pipeline-review]] — Pipeline 视角的 STFM 综述，提供更深入的方法论分类
- [[source-cast]] — CaST 因果时空预测，本文在因果推理方向的引用

[^src-st-foundation-models-survey]: [[source-st-foundation-models-survey]]
