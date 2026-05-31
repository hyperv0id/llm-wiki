---
title: "MTGNN"
type: entity
tags:
  - time-series
  - multivariate
  - graph-neural-network
  - spatial-temporal
  - forecasting
  - KDD-2020
created: 2026-05-30
last_updated: 2026-05-30
source_count: 2
confidence: high
status: active
---

# MTGNN

MTGNN (Multivariate Time Series Forecasting with Graph Neural Networks) 是首个从图神经网络 (GNN) 视角出发的多变量时间序列预测通用框架，由 Wu, Pan, Long 等人提出（KDD 2020）[^src-mtgnn]。核心创新在于*无需预定义图结构*——通过自适应图学习模块从数据中提取变量间隐含的单向依赖关系。

## 问题设定

给定 N 个变量的 P 步历史观测，预测未来 Q 步值。变量之间可能存在潜在依赖（如道路 A 的拥堵影响道路 B），但该依赖关系未知且可能非对称 [^src-mtgnn]。

## 架构

MTGNN 由四个核心模块交替堆叠构成 [^src-mtgnn]：

1. **[[graph-learning-layer|图学习层]]** — 通过两组可学习节点嵌入 E₁、E₂ 计算单向邻接矩阵 A = ReLU(tanh(α(M₁M₂ᵀ − M₂M₁ᵀ)))。每个节点保留 top-k 最近邻以保持稀疏性。可融合外部静态特征作为节点嵌入 [^src-mtgnn]。

2. **[[mix-hop-propagation-layer|Mix-Hop 传播层]]** — 两步操作：信息传播步骤保留部分根节点原始状态以避免过平滑；信息选择步骤汇总各跳信息并通过可学习参数过滤噪声。图卷积模块使用两个 mix-hop 层分别处理入流/出流信息，求和得到净流入信息 [^src-mtgnn]。

3. **[[dilated-inception-layer|扩张初始层]]** — 四个滤波器尺寸 (1×2, 1×3, 1×6, 1×7) 覆盖常见周期模式；扩张因子指数增长使感受野呈指数级扩展。时间卷积模块使用双通道（一个 tanh 滤波 + 一个 sigmoid 门控）[^src-mtgnn]。

4. **输出模块** — 两层 1×1 标准卷积，将通道维度映射到期望输出维度（单步预测 = 1，多步预测 = Q）[^src-mtgnn]。

残差连接从时间卷积模块输入跳跃到图卷积模块输出；跳连接在每个时间卷积模块后汇聚到输出模块 [^src-mtgnn]。

## 训练策略

- **子图采样**：每轮迭代随机分组节点，图学习层复杂度从 O(N²) 降至 O((N/s)²)，解决大图内存瓶颈 [^src-mtgnn]
- **课程学习**：逐步增加预测步长——从预测 1 步开始，逐渐过渡到完整预测窗口，帮助模型找到更好的局部最优 [^src-mtgnn]

## 性能

| 任务 | 数据集 | 节点数 | 表现 |
|------|--------|--------|------|
| 单步预测 | Solar-Energy | 137 | SOTA |
| 单步预测 | Traffic | 862 | RSE 显著优于所有基线（horizon 3: -7.24%） |
| 单步预测 | Electricity | 321 | SOTA |
| 单步预测 | Exchange-Rate | 8 | **失败**（小图 + 少样本） |
| 多步预测 | METR-LA | 207 | 与 DCRNN/STGCN/GWN/GMAN 持平（*无需预定义图*） |
| 多步预测 | PEMS-BAY | 325 | 与 GWN/GMAN 持平（*无需预定义图*） |

在多步预测中，MTGNN 不使用道路网络拓扑即达到与依赖预定义图的 STGNN 模型持平的性能，证明了自适应图学习的有效性 [^src-mtgnn]。

## 意义与局限

MTGNN 是为数不多的不依赖预定义图结构即可建模多变量时间序列的 GNN 框架，与同团队 Graph WaveNet (IJCAI 2019) 共同开创了自适应图学习范式 [^src-mtgnn]。它是 [[cross-dimension-dependency|跨维度依赖]] GNN 路线的起点，后续被 [[crossformer|Crossformer]]（ICLR 2023）等作为基线对比 [^src-crossformer-2023]。

局限包括：在极小图（< 10 节点）上失效；学习的图是静态的（不随输入时序变化）；在大规模图上推理仍需 O(N²) 构建全图 [^src-mtgnn]。

## 相关页面

- [[source-mtgnn]] — 源文件摘要
- [[cross-dimension-dependency]] — 跨维度依赖概念
- [[traffic-forecasting]] — 交通预测应用场景
- [[graph-learning-layer]] — 图学习层技术
- [[mix-hop-propagation-layer]] — Mix-Hop 传播层技术
- [[dilated-inception-layer]] — 扩张初始层技术
- [[crossformer]] — 后续 Transformer 建模 CD（将 MTGNN 作为基线）
- [[source-crossformer-2023]] — Crossformer 源文件

[^src-mtgnn]: [[source-mtgnn]]
[^src-crossformer-2023]: [[source-crossformer-2023]]
