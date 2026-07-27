---
title: "STUNet: Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - spatio-temporal
  - generalization
  - zero-shot
  - transformer
  - explicit-graph-modeling
  - tokenization
created: 2026-07-27
last_updated: 2026-07-27
source_count: 0
confidence: low
status: active
---

# STUNet: Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting

**Authors:** Yujun Chen\*, Shihao Tu\*, Wenyue Ding, Yicheng Lu, Qingkai Ren, Yangjie Zheng, Yang Yang† (Zhejiang University / SUPCON Technology) · **Venue:** KDD 2026 (Jeju), DOI 10.1145/3770855.3817746 · **Code:** `github.com/JimmyChen6/STUNet` · **raw:** `raw/stunet-unified-spatio-temporal-tokens-generalizable-traffic-forecasting.pdf`

## 核心问题（叙事主线：跨网络泛化）

STGNNs 与 spatio-temporal Transformer 虽在单网路上表现强，但**隐式建模空间结构**：图消息传递或 attention 分数把空间关系与传感器时序观测耦合。训练时时间波动会污染“本应由道路拓扑决定”的空间表示，换一张路网（传感器数、邻接、地理区域都不同）就失效。节点级 embedding 的 graph-free 方法（如 STID）换场景必须重训 embedding，同样缺跨网络零样本能力。论文要把**空间结构显式 token 化**，与时间解耦，做成可跨网络复用的表示基。

## 方法

**[[stunet|STUNet]]（Spatio-Temporal Unified Network）** 四件套 + 两阶段训练：

1. **[[spatial-tokenizer-adjacency-patches|Spatial tokenizer]]** — 将关系图邻接矩阵 $A\in\mathbb{R}^{N\times N}$ 切成固定大小非重叠 patch，MLP 映到统一维度的 spatial tokens $E_s$。传感器数可变时靠 zero-pad；patch 化避免点级 $O(N^2)$ 爆炸，并保留结构信息（ViT 式启发，对象是**邻接矩阵**而非地理坐标点）。
2. **Temporal tokenizer** — 沿时间对每个传感器做 PatchTST 式切块，拼 day-of-week / time-of-day embedding，得到 temporal tokens $E_t$。
3. **[[query-aggregate-attention|Query-Aggregate Attention]]** — 因两类 token 语义与位置语义不对齐，禁止朴素 full attention。Query 阶段：temporal 作 Q、spatial 作 K/V，用两套 RoPE 让传感器在邻接矩阵的行/列相对位置上定位上下游；Aggregate 阶段：在融合后的 temporal 表示上自注意力，同时聚合“相关传感器”与“时间滞后”。空间 token 跨层复用。
4. **Projection head** — 仅用融合后的 temporal 输出做 $H{=}12\to F{=}12$ 预测；损失为预测误差范数。

**两阶段训练（泛化的关键设计）：** Stage 1 分别用 autoencoder 预训练两 tokenizer；Stage 1 对 spatial tokenizer **随机打乱节点索引** 生成多样邻接，增强结构鲁棒。Stage 2 接入 backbone 时 **冻结 spatial tokenizer**，只用预测损失更新其余参数——保证空间表示不被时间信号回写。

## 实验重点：零样本跨网络（RQ1）

协议：在互不重叠的流数据集 SD / GBA / GLA（LargeST）与速度数据集 METR-LA / PEMS-BAY / SZ-TAXI 上，**在一个网络收敛后直接测另外两个**（graph-free 基线去掉 node-wise embedding）。对比 DLinear、PatchTST、STID、iTransformer、STGCN、STWave、[[patchstg|PatchSTG]]。论文结论：

- 多变量（带空间）整体强于单变量；
- **显式空间建模在全部迁移对与指标上最优**——相对隐式图/attention 与 node embedding 路线。

示例（Table 1 打印的跨区零样本平均 horizon，SD→GBA）：STUNet MAE **34.46** / RMSE **53.90**，PatchSTG 37.98 / 59.35，STGCN 38.58 / 60.37。

In-domain（RQ2，LargeST 等）：与 D2STGNN、PatchSTG、STID 等可比；**RMSE 常最优**，作者解读为高峰大值更准、对调度更有用。消融（Table 4）：去掉 spatial tokens 跌幅最大；去掉 pretrain / freeze、改 full attention、去掉邻接置换增强、temporal 预训练改 reconstruction 均变差。附录 UMAP：直路 / Y 形 / 环状结构的 spatial token 聚类 ARI 0.96、NMI 0.98。

## 局限与定位

- 不是多城市大规模预训练的 [[spatio-temporal-foundation-model|ST foundation model]]：单源训练 + 跨网 zero-shot，靠**显式结构 token + 冻结**，不是 prompt/memory 库。
- 仍依赖给定关系图邻接；动态图/事故冲击未作为主设定。
- 与 [[stop|STOP]] 同攻结构泛化，但路线相反：STOP **切断** node-to-node messaging；STUNet **显式编码** 邻接并 query 上下游。
- 效率表显示大图可训，但非纯线性 $O(N)$ 叙事主角；主卖点是泛化而非吞吐。

## 相关页面

- [[stunet]] — 模型主页
- [[spatial-tokenizer-adjacency-patches]] — 邻接矩阵 patch 空间 tokenizer
- [[query-aggregate-attention]] — 时空融合机制
- [[traffic-forecasting]] · [[ood-generalization]] · [[spatio-temporal-ood-learning]] · [[patchstg]]
