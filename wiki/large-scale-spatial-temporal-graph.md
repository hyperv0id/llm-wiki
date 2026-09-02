---
title: "Large-Scale Spatial-Temporal Graph Forecasting"
type: concept
tags:
  - spatial-temporal
  - computational-complexity
  - scalability
created: 2026-04-29
last_updated: 2026-08-30
source_count: 8
references:
  - [[source-fast-long-horizon-forecasting]]
confidence: high
status: active
---

# Large-Scale Spatial-Temporal Graph Forecasting

## 定义

大规模时空图预测指在具有数千至数万个节点（图结构传感器/区域）的网络上进行时间序列预测，通常需要**长视野**（多步至一周）预测。

典型场景：
- **交通网络**：加州高速网络 8600+ 检测器
- **能源网络**：城市级电表/燃气表
- **空气监测网络**：跨城市的传感器阵列

## 核心挑战

### 1. 计算复杂度

| 模块 | 复杂度 | 大规模场景问题 |
|------|---------|-----------------|
| 空间卷积 (GCN) | O(N²) | 8600 节点 = 74M 边计算 |
| 时间注意力 | O(T²) | 672 步 = 451k 计算 |
| 联合时空 | O(N²T + NT²) | 内存爆炸 |

### 2. 内存消耗

现有 STGNN 在 CA 数据集（8600 节点，672 步预测）上需要 **48GB GPU 内存**，大多数 GPU 无法训练。

### 3. 长视野预测精度退化

预测步数增加时：
- 错误累积传播
- 空间依赖衰减
- 时序模式复杂化

## FaST 的解决方案

FaST 提出两个核心技术：

1. **[[adaptive-graph-agent-attention|AGA-Att]]**: O(N·a) 空间复杂度（a = 32 ≪ N）
2. **[[mixture-of-experts|HA-MoE]]**: 时间压缩输入避免 T 步展开

## 基础模型方法

[[most|MoST]] (KDD 2026) 采用零样本基础模型范式处理大规模时空图：在多个城市数据集上预训练后，无需微调即可在新城市上预测[^src-most]。通过仅建模 top-k 最近邻的空间交互（而非全图），将空间复杂度从 O(N²) 降至 O(N·k)，同时利用多模态背景信息提升泛化能力[^src-most]。在 SD (716节点)、GBA (2,352节点)、GLA (3,834节点) 上零样本性能超越多数全量训练模型[^src-most]。

## 现有方法分类

### 结构感知方法
- **稀疏聚合** (SGP): 利用图拓扑减少计算
- **邻居采样** (SAGDFN): 随机采样邻居
- **图划分 / 空间分块 ([[patchstg|PatchSTG]])**: 使用 [[leaf-kdtree|leaf KDTree]] 将不规则分布的交通点划分为平衡、不重叠的 patches，在 patch 内做 depth attention（局部空间），在 patch 间同索引位置做 breadth attention（全局空间）。复杂度 O(NRd) 但无信息损失，兼具可解释性和保真性。CA 数据集 (8,600 节点) 上实现 10× 训练加速和 4× 内存节省[^src-patchstg]。

局限：依赖准确图结构、丢弃长程依赖

- **可学习子图划分 ([[lets-group|Let's Group]] SGL, IJCAI 2025)**: 与静态划分相对，用 M 个可学习记忆向量按特征相似度选 top-K 节点构成可重叠子图，子图内共享同一图构造函数建 K×K 相关矩阵，跨子图特征平均聚合；复杂度 O(N²)→O(N+K²)（该文 Sec. 3.6）。作者报告在 PEMS03/04/07/08 上为 8 个 backbone 挂载 SGL 后性能相当，平均 GPU 内存最高降 56.4%（摘要，未说明平均口径），DDGCRN 在 883 节点的 PEMS07 上降 60.5%（Sec. 4.1, Table 3）[^src-lets-group]。

### 无结构方法
- **线性注意力** ([[bigst|BigST]]): 用正随机特征 (PRF) 核近似 + [[linearized-spatial-convolution|线性化空间卷积]]降到 O(N)，绕过成对节点计算[^src-bigst]
- **图传播 + 统一时空线性注意力 ([[stgformer|STGformer]], arXiv 2024)**: SGC 式图传播保留 0..k 阶输出，空间/时间共享同一组 QKV 并线性化为分解内积（O(N+T)），单层替代堆叠注意力；论文报告 8600 节点 CA 图批量推理较 STAEformer 100× 加速、99.8% GPU 内存降，该设置下 FLOPs 比值约 0.00131（T=12、d=32、K=3、L=3，Sec IV-D）[^src-stgformer]
- **随机投影** (RPMixer): MLPs 融合空间特征
- **空间标识** (STID): 消除空间交互

局限：过度简化交互、引入噪声、丢失空间语义

### 余弦相似度线性化方法
- **[[efficient-cosine-operator|ECO]]** (RAGC, 2026): 基于余弦相似度的图卷积算子，通过矩阵结合律将 O(N²) 复杂度降至 O(N)，避免近似噪声

与 [[bigst|BigST]] 的随机特征映射近似 softmax 不同，ECO 直接使用余弦相似度度量节点相似性，无需非线性激活，因此不引入近似噪声。通过门控机制（softmax + ReLU 逐元素乘积）确保邻接矩阵非负性和稀疏性[^src-ragc-efficient-traffic-forecasting]。

### 异质性感知方法
- **HA-MoE**: 动态选择专家适配节点和时间异质性
- **代理注意力**: 用少量代理 token 捕获空间冗余

## 数据集基准

LargeST 数据集（KDD 2024）：
| 数据集 | 节点数 | 时间粒度 | 时间跨度 |
|--------|--------|----------|----------|
| SD | 716 | 15min | 2019全年 |
| GBA | 2,352 | 15min | 2019全年 |
| GLA | 3,834 | 15min | 2019全年 |
| CA | 8,600 | 15min | 2019全年 |

XTraffic 数据集（2024）：基于加州 2023 年交通数据，包含时间对齐的事件记录，用于事件引导的时空预测[^src-incident-guided-st-forecasting]：
| 子数据集 | 节点数 | 边数 | 事件数 |
|----------|--------|------|--------|
| Alameda | 521 | 13,828 | 14,687 |
| Contra Costa | 496 | 13,339 | 5,587 |
| Orange | 990 | 29,142 | 18,700 |

此外，[[virtual-nodes-traffic|虚拟节点]]长期预测工作（arXiv 2025）以 5 分钟采样使用其中最小的 SD 子集（716 节点、17,319 边、密度 0.0338，Table 1；数据集时间跨度为 2017-01-01 至 2021-12-31，实验训练/测试实际使用其中 2019-2020 年数据），作者报告在该设置下 semi-adaptive 虚拟节点将 STGCN 的 75-100 分钟平均 RMSE 从 45.15 降至 42.32[^src-virtual-nodes]。

## 相关页面

- [[traffic-forecasting]] — 城市交通预测场景
- [[most]] — 多模态零样本基础模型方法
- [[adaptive-graph-agent-attention|AGA-Att]] — 空间复杂度优化
- [[mixture-of-experts|MoE]] — 特征提取与时间压缩
- [[ragc]] — 正则化自适应图卷积方法
- [[patchstg]] — KDTree 空间分块 + 双注意力高效建模
- [[leaf-kdtree]] — 不规则空间数据的平衡分块算法
- [[irregular-spatial-patching]] — 三步空间分块管道
- [[efficient-cosine-operator|ECO]] — 余弦相似度线性复杂度图卷积
- [[bigst]] — 线性注意力 (PRF 核) 的大规模 STGNN，可扩展到 ~10 万节点
- [[lets-group]] — 记忆向量锚点的可学习子图划分，建模层面降内存 (IJCAI 2025)
- [[stgformer]] — 单层图传播 × 统一时空线性注意力的高效交通 Transformer (arXiv 2024)
- [[stg-attention]] — STG-Attention 机制页
- [[virtual-nodes-traffic]] — 虚拟节点图改写：以消息传递路径缩短缓解 over-squashing 的长期预测 (arXiv 2025)
- [[graph-node-clustering]] — 图节点聚类方案全景：拓扑/特征/可学习三类聚类与静态 vs 动态取舍 (analysis)

[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
[^src-most]: [[source-most]]
[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
[^src-patchstg]: [[source-patchstg]]
[^src-bigst]: [[source-bigst]]
[^src-lets-group]: [[source-lets-group]]
[^src-stgformer]: [[source-stgformer]]
[^src-virtual-nodes]: [[source-virtual-nodes]]