---
title: "图节点聚类方案全景：拓扑驱动、特征驱动与可学习聚类"
type: analysis
tags:
  - clustering
  - graph
  - spatial-temporal
  - graph-partitioning
  - community-detection
  - analysis
created: 2026-09-01
last_updated: 2026-09-02
source_count: 13
confidence: high
status: active
---

# 图节点聚类方案全景：拓扑驱动、特征驱动与可学习聚类

本文综合 wiki 中 12 个源文件里出现的图节点聚类做法，按「聚类依据」归为三类：**拓扑结构驱动**（经典社区发现/图划分）、**特征/相似度驱动**（表示空间聚类）、**可学习/端到端聚类**（随训练优化）。另按用途给出选择指南与静态 vs 动态的取舍。

## 问题：为什么 ST 任务需要节点聚类

时空图预测/插补中节点数 N 常达数千（CA 数据集 8,600 节点[^src-patchstg]），全图 O(N²) 相关矩阵成为计算与内存瓶颈[^src-lets-group]；同时节点间存在空间冗余——共享模式可以用少量潜在原型近似节点表示流形[^src-gpt-st]。聚类由此承担两类职能：**降复杂度**（把全图切成子图）与**发现语义结构**（区域类型、行为模式）。

## 一、拓扑结构驱动（经典社区发现 / 图划分）

- **Louvain 社区检测**（模块度贪心优化）。FCGCN 按拓扑关系划分子图[^src-lets-group]；GAMMA-Net 用它做可解释性分析，验证空间注意力忠实反映地理邻近性、时间注意力揭示功能性连接（行为相似但地理位置遥远的传感器）[^src-gamma-net]。
- **METIS 平衡图划分**（最小化割边 + 子图规模平衡）。LarSTL 用它加地理坐标做持续学习子图划分[^src-lets-group]；UniFlow 用它把 graph 数据分割为平衡子图做 mean pooling，实现网格+图数据的统一 patch 化[^src-uniflow]；Let's Group 消融中作为静态划分基线（PEMS04 MAE 18.71 vs 学习式 SGPM 18.49）[^src-lets-group]。
- **KD-Tree 地理分块**。PatchSTG 用 leaf KDTree 按地理坐标切分平衡、不重叠的 patches（CA 8,600 节点上 10× 训练加速、4× 内存节省）；KDTree 满足平衡+不重叠保证，METIS 因递归合并语义不同、KMeans 因簇极不平衡（max/min 比高达 96:6）均失败[^src-patchstg]。
- **结构熵最小化编码树（deDoc 式贪心）**。MultiSPANS 用 combination/merge 树算子贪心最小化图结构熵，得到层次化社区结构（编码树）：每层对应一个图划分，直接作为一层注意力掩码，树的深度在优化中自适应确定（论文称取决于图的规模与结构复杂度）；消融中该掩码优于用最小熵式层次社区检测 Infomap 构造的掩码（PEMSD4-flow MAE 19.43 vs 19.07）[^src-multispans]。与 Louvain/METIS 的单层划分不同，编码树一次给出多粒度层级（见 [[structural-entropy]]）。

> [!note] 通用知识补充（课程层面，wiki 语料未覆盖）
> 经典社区发现还有谱聚类（拉普拉斯特征分解 + 低维 k-means）、标签传播、Girvan-Newman 边介数删除等，均基于拓扑结构而非节点特征。以下论述仅来自本课程组织，无 wiki 源文件支持。

## 二、特征 / 相似度驱动（表示空间聚类）

- **k-means on embeddings**。Moirai-MoE 对预训练 dense 模型的 token 嵌入做 mini-batch k-means，提取每层簇中心 $C^l$，token 到簇中心的欧氏距离作为 token-to-expert 亲和度引导专家路由[^src-moirai-moe]；PatchSTG 消融中 KMeans 因簇不平衡无法满足双注意力对平衡 patch 的要求[^src-patchstg]。
- **k-medoids**。PFRP 的全局记忆库 GMB 用它把训练样本压缩为 K 个 medoid——中心必须是真实样本而非合成均值，保证检索到的模式是真实连贯的历史序列[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。
- **DTW 相似度 + METIS**。Let's Group 消融的静态划分对照组之一（-METIS），固定划分、不随节点相关性动态变化[^src-lets-group]。
- **KD-Tree 贪心容量约束聚类**。UrbanFM 的 MiniST 用它将异构传感器布局转化为统一可学习 token，无需邻接矩阵、token 间结构独立可并行训练，本地聚合隐式编码"近者更相关"的地学原理[^src-urbanfm]。
- **随机划分**。VisiFold 把剩余节点随机切分为固定大小子图以提升并行度、兼作隐式正则化，是最朴素的对照[^src-lets-group]。

## 三、可学习 / 端到端聚类（随训练优化）

- **胶囊动态路由聚类（GPT-ST）**。区域 embedding 经 squash 归一化 + 2 轮动态路由迭代"投票"产生 H_S=10 个语义簇中心（商业区、住宅区等），上层空间超图（H_M=16）再学簇间迁移模式（如"住宅→商业"通勤流）[^src-gpt-st]。论文自述局限包括固定簇数跨城市不现实[^src-gpt-st]。
- **对比聚类（MiniTraffic）**。InfoNCE 损失学 patch 间余弦相似度，构建 k-NN 稀疏图——按语义相似度（而非空间邻近性）动态建图，图注意力复杂度从 O(N²) 降至 O(k·N)[^src-minitraffic]。
- **记忆向量锚点子图划分（Let's Group SGPM）**。M 个可学习记忆向量作锚点，按特征相似度选 top-K 节点构成可重叠子图，SGFAM 跨子图平均聚合去重叠冗余；相关矩阵从 N×N 降为 K×K，复杂度 O(N²)→O(N+K²)[^src-lets-group]。
- **注意力分数动态聚类（FENCE）**。去噪每步对空间注意力分数 A∈ℝ^{N×N} 做 k-means，按聚类内对数后验均值共享引导尺度，解决固定 CFG 尺度在低条件信息场景的漂移[^src-fence]。
- **无监督涌现聚类（STBP 模式库）**。无显式聚类约束与监督下，模式库自主把节点组织为周期/趋势行为相似的簇（同一簇内相似、簇间异质），且新节点被正确归入既有簇——论文归因于预测任务驱动的自主学习[^src-stbp]。
- **可学习原型（ST-SSDL）**。prototype triplet loss 将潜在空间离散化为历史锚点原型，作为自监督偏差学习的基础[^src-st-ssdl]。

## 四、按用途选择

| 用途 | 方案 | 来源 |
|---|---|---|
| 可扩展性（打破 O(N²)、并行） | KD-Tree[^src-patchstg][^src-urbanfm]、METIS[^src-uniflow]、随机划分[^src-lets-group]、SGPM[^src-lets-group] | 一、二、三 |
| 语义区域发现 | GPT-ST 胶囊聚类[^src-gpt-st] | 三 |
| 图结构学习（无先验邻接） | MiniTraffic 对比聚类 kNN 图[^src-minitraffic] | 三 |
| 检索库压缩 | PFRP k-medoids[^src-predicting-the-future-by-retrieving-the-past-aaai2026] | 二 |
| 引导/正则 | FENCE 聚类级引导[^src-fence]、VisiFold 随机划分[^src-lets-group] | 一、三 |

## 取舍：静态 vs 动态

静态结构划分（Louvain/METIS/KD-Tree）简单、可解释、无训练开销，但子图固定、不随数据动态变化——这是 Let's Group 对静态划分一族的主要批评（将 PatchSTG 地理坐标划分与 FCGCN/Louvain、LarSTL/METIS 并列）[^src-lets-group]。学习式聚类能捕捉动态语义依赖（GPT-ST 的簇间迁移[^src-gpt-st]、FENCE 的逐去噪步重聚类[^src-fence]），但引入额外参数与训练开销，且需注意固定簇数的现实性局限[^src-gpt-st]。特征驱动聚类居中间：KMeans 类方法简单但簇质量受数据分布影响（PatchSTG 的不平衡问题[^src-patchstg]），k-medoids 以真实性换计算量[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。

## 相关页面

- [[lets-group]] — 学习式记忆向量子图划分 + 静态划分消融对照 (IJCAI 2025)
- [[patchstg]] — leaf KDTree 地理分块 (KDD 2025)
- [[gpt-st]] — 胶囊动态路由语义聚类预训练 (NeurIPS 2023)
- [[minitraffic]] — 对比聚类 kNN 稀疏图 (ICML 2026)
- [[fence]] — 注意力分数 k-means 聚类级引导 (AAAI 2026)
- [[cluster-aware-guidance]] — FENCE 的聚类感知引导技术页
- [[cluster-based-gating]] — Moirai-MoE 的 k-means 簇中心门控
- [[k-medoids-clustering]] — K-medoids 聚类技术页
- [[global-memory-bank]] — PFRP 的 K-medoids 记忆库
- [[urbanfm]] — MiniST KD-Tree 聚类 token 化
- [[uniflow]] — METIS 图分割统一 patch 化
- [[contextual-pattern-bank]] — STBP 涌现聚类模式库
- [[ssdl]] — ST-SSDL 原型离散化
- [[traffic-forecasting]] — 交通预测任务总览
- [[multispans]] — 结构熵编码树做多层注意力掩码的模型页
- [[structural-entropy]] — 结构熵与编码树概念页
- [[large-scale-spatial-temporal-graph]] — 大规模时空图的可扩展方法总览

[^src-lets-group]: [[source-lets-group]]
[^src-gamma-net]: [[source-gamma-net]]
[^src-uniflow]: [[source-uniflow]]
[^src-patchstg]: [[source-patchstg]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
[^src-urbanfm]: [[source-urbanfm]]
[^src-gpt-st]: [[source-gpt-st]]
[^src-minitraffic]: [[source-minitraffic]]
[^src-fence]: [[source-fence]]
[^src-stbp]: [[source-stbp]]
[^src-st-ssdl]: [[source-st-ssdl]]
[^src-multispans]: [[source-multispans]]
