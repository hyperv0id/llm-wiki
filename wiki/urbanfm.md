---
title: "UrbanFM"
type: entity
tags:
  - foundation-model
  - spatio-temporal
  - transformer
  - scaling-laws
  - zero-shot
  - traffic-forecasting
  - imputation
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# UrbanFM

**UrbanFM** 是一个以 scaling 为核心方法论的城市时空基础模型（arXiv 2026），由 HKUST(GZ) / HKUST 的 Wei Chen, Yuqian Wu, Junle Chen, Xiaofang Zhou, Yuxuan Liang 提出[^src-urbanfm]。UrbanFM 首次从第一性原理出发，系统性地将城市时空基础模型扩展分解为三个相互配合的维度：数据规模化（WorldST）、计算规模化（MiniST）、架构规模化（UrbanFM 模型本体），在零样本泛化、跨任务迁移和鲁棒性上全面超越现有基础模型和专职专家模型[^src-urbanfm]。

## 核心理念

UrbanFM 受 Sutton 的"苦涩教训"启发——当计算规模超越关键阈值时，利用海量数据的通用范式必然优于复杂的人工启发式设计[^src-urbanfm]。因此，UrbanFM 摒弃了现有模型大量注入的手工先验（如辅助图结构、节点 prompt），采用极简 Transformer 架构，仅依赖注意力机制从海量数据中自主学习动态依赖[^src-urbanfm]。

## 三大组件

### 1. WorldST — 数据规模化

十亿级规模的全球城市语料库，覆盖 100+ 城市 × 8 个领域 × 1 亿+数据点。三阶段流水线：多源采集（开放政府 + 学术仓库 + PeMS API）→ 统一摄入（格式统一 + 频率同步到 5 分钟）→ 质量控制（死传感器剔除、3σ 截断、线性插值预填补）[^src-urbanfm]。在空间和时间覆盖上远超 [[unist|UniST]]、[[opencity|OpenCity]] 和 BigCity 达 33–145 倍[^src-urbanfm]。

### 2. MiniST — 计算规模化

城市数据输入维度因城市而异（传感器数 Nx ≠ Ny），MiniST 通过 KD-Tree 贪心容量约束聚类将异构空间结构转化为统一的可学习 token（patch 尺寸 Sp×Tp）[^src-urbanfm]。关键优势：无需邻接矩阵、token 间结构独立可实现大规模并行训练、本地聚合隐式编码"近者更相关"的地学原理[^src-urbanfm]。

### 3. UrbanFM 架构 — 架构规模化

极简 Transformer，默认 8 层（4 spatial + 4 temporal）[^src-urbanfm]：

| 组件 | 设计 | 作用 |
|------|------|------|
| 分解时空注意力 | 先 temporal（节点内）再 spatial（节点间）self-attention | 点级细粒度动态建模，避免展平序列的 O((NT)²) 复杂度 |
| ST-RoPE | T-RoPE（相对时间距离）+ S-RoPE（相对空间序） | 编码时空位置关系，支持变长上下文外推，无需邻接矩阵 |
| RevIN + 生成式目标 | 实例归一化 + 未来时间步零噪声 mask | 处理非平稳性 + 统一 forecasting 和 imputation 为单一重建任务 |

极简设计还使 UrbanFM 可直接利用 Flash Attention 和 Linear Attention 等硬件优化实现大规模扩展[^src-urbanfm]。

## 关键性能

在含 12 数据集、22 baseline 的 EvalST 基准上[^src-urbanfm]：

- **零样本**：MAPE 优于现有时空基础模型 39.0%–70.2%，超越部分 full-shot 专职模型（如长时传感器预测 UrbanFM 17.0 MAPE vs D2STGNN 20.5）
- **小样本微调**：仅 10% 目标域样本提升 28.2%–65.2% 超越 full-shot 专家模型
- **跨任务**：预训练阶段无 imputation 目标，零样本填补仍最优，显著超越 MICE/SVD
- **Scaling 律**：模型深度 ↑ → 误差 ↓（2→8 层持续递减）；数据比例 ↑ → MAPE 幂律衰减，未饱和
- **鲁棒性**：30% 零掩码+噪声注入下保持最高保真度；生成式预训练将信号恢复作为核心目标
- **效率**：推理约 Chronos 的 1/4、TimeMoE 的 1/10（A100），MAPE 最优 15.2%

## 与现有基础模型的对比

| 模型 | 数据类型 | 架构 | 零样本 | 核心特点 |
|------|---------|------|--------|---------|
| UrbanFM | Sensor + Grid | 极简 Transformer（因子分解注意力+ST-RoPE） | ✓ | 以 scaling 为中心；WorldST+MiniST+极简架构 |
| [[urbandit|UrbanDiT]] | Sensor + Grid | Diffusion Transformer + 统一 prompt | ✓ | 扩散范式+三 memory pool |
| [[uniflow|UniFlow]] | Sensor + Grid | Encoder-Decoder + ST-MRA | ✓ | 统一 grid+graph，4 memory pools |
| [[unist|UniST]] | Grid only | MAE + memory prompt | ✓ | 首个 one-for-all 网格基础模型 |
| [[opencity|OpenCity]] | Sensor only | TimeShift Transformer + GNN | ✓ | 开源，3 种规模 |
| [[urbangpt|UrbanGPT]] | Grid only | LLM (Vicuna-7B) + instruction-tuning | ✓ | LLM-based，7B 参数 |

## 与其他页面的链接

- [[spatio-temporal-foundation-model]] — 时空基础模型总览
- [[traffic-forecasting]] — 交通预测通用背景
- [[factost]] — FactoST, 同实验室 (Yuxuan Liang) 的因子化时空基础模型，UTP+STA 两阶段设计
- [[unist]] — 同实验室（Tsinghua FIB lab 部分作者转移至 HKUST-GZ）
- [[uniflow]] — 统一 grid+graph 基础模型
- [[urbandit]] — 扩散 Transformer 基础模型
- [[opencity]] — 开源时空基础模型
- [[urbangpt]] — 基于 LLM 的时空基础模型
- [[urbanpg]] — UrbanPG, prompt-backbone 解耦实现大规模+小样本+持续学习三合一 (AAAI 2026)
- [[urbanverse]] — UrbanVerse, 互补方向——区域属性预测基础模型（犯罪/人口/碳排放），跨城市跨任务范式 (arXiv 2026)

[^src-urbanfm]: [[source-urbanfm]]
