---
title: "Source: PAST — Primary-Auxiliary Spatio-Temporal Network for Traffic Time Series Imputation"
type: source-summary
tags:
  - traffic-imputation
  - spatio-temporal
  - graph-neural-network
  - pvldb
  - missing-data
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: medium
status: active
---

# Source: PAST — A Primary-Auxiliary Spatio-Temporal Network for Traffic Time Series Imputation

**作者**：Hanwen Hu, Zimo Wen, Shiyou Qian, Jian Cao (Shanghai Jiao Tong University)
**发表**：PVLDB, arXiv:2511.13414 (2025-11-17)
**代码**：github.com/Hanwen-Hu/PAST

## 核心贡献

PAST 将交通时间序列的潜在模式划分为两类：**主模式（primary patterns）**（来自数据点内部关系）和**辅助模式（auxiliary patterns）**（来自时间戳、节点属性等外部因素），通过双模块架构分别建模，并在三个缺失类型（随机/Random、纤维/Fiber、块/Block）下统一处理填补[^src-past]。

主模式感知局部时序波动和邻近拓扑依赖，辅助模式通过外部信息高效捕获长期周期性和大范围空间相似性。二者存在与缺失类型的自然对应：随机缺失主要依赖主模式，纤维/块缺失需要辅助模式增强。

## 架构概要

PAST 采用双模块架构：

1. **Graph-Integrated Module (GIM)**：纯 GNN，捕获主模式。将不完整时间序列建模为动态有向图——缺失值对应节点/边的缺失。核心设计：
   - **Temporal Layer**：构造有向时间图 $G_T$，观测值之间双向全连接，观测→缺失单向传播，缺失之间无边。应用 interval-aware dropout（近邻边更高概率丢弃，迫使模型学习更广范围依赖）。外部信息通过共享隐向量注入
   - **Spatial Layer**：基于交通网络拓扑邻接矩阵 $A_S$ 建模节点间依赖，使用 multi-order convolution（将 $A_S$ 升幂到 $K$ 阶）捕获多跳空间依赖
   - 层内以时间层→空间层的顺序堆叠

2. **Cross-Gated Module (CGM)**：提取辅助模式。输入为外部空间嵌入（节点身份）和时间嵌入（周/时/分时间戳），通过 cross-gated layer 处理：
   - 四个 $d \times d$ 线性子层（空间投影/门控、时间投影/门控），而非简单拼接
   - 门控向量经 sigmoid 筛选本域特征，经 tanh 建模跨域正负交互
   - 公式 $\boldsymbol{v}_{sp} \leftarrow \boldsymbol{v}_{sp} \cdot \text{Sigmoid}(\boldsymbol{v}_{sg}) \cdot \text{Tanh}(\boldsymbol{v}_{tg})$
   - 各层将隐向量前传至对应 GIM 层进行跨模块信息交换

3. **Ensemble 训练框架**：受 GBDT 启发，GIM 通过观测值 MSE 自监督训练（Loss 1），CGM 最小化其输出与 GIM 训练残差的 MSE（Loss 2），使 CGM 作为辅助增强器捕获 GIM 遗漏的模式

最终填补：$\boldsymbol{Y} = M \odot X + (1-M) \odot (Y_{CGM} + Y_{GIM})$

## 实验结果

在 METR-LA、PeMS-Bay、LargeST-SD 三个数据集上，覆盖 27 种缺失条件（随机缺失 3 种缺失率、纤维缺失 3 种长度、块缺失 3 种长度×跨度组合），与 7 个基线对比：

- **随机缺失**：RMSE 平均降低 8.2%，MAE 降低 11.2%
- **纤维缺失（r=0.4）**：离线 RMSE 降低 12.8%，MAE 降低 10.1%；在线 RMSE 降低 18.2%，MAE 降低 19.4%
- **块缺失（r=0.4）**：离线 RMSE 降低 14.4%，MAE 降低 10.9%；在线 RMSE 降低 **26.2%**，MAE 降低 **31.6%**

PAST 在结构化缺失（纤维/块）场景下优势最大，因其 primary-auxiliary 范式显式对齐了模式提取与具体缺失类型。消融实验验证了 GIM（移除后 RMSE 上升 9.4%）、CGM（移除后 RMSE 上升 7.2%）、interval-aware dropout 和 cross-gated 机制各自的贡献。

## 局限性

1. GIM 时间复杂度 $O(nT^2 N d^2)$，长序列下时序图构建开销较高
2. 仅在三个美国交通数据集上评估，地理泛化性未验证
3. 未来方向：多模态数据（文本、事故记录等）的整合

## 历史定位

PAST 在填补模型谱系中的独特位置：
- **vs GRIN (ICLR 2022)**：GRIN 使用双向 MPGRU 从内部数据关系建模，未利用外部信息；PAST 通过 CGM 显式引入外部特征增强纤维/块缺失填补
- **vs CSDI (NeurIPS 2021)**：CSDI 是扩散模型用于一般时间序列填补，仅处理随机缺失；PAST 针对交通场景的三类缺失统一建模
- **vs STCPA (CIKM 2022)**：STCPA 是交通专用模型但仅基于内部模式；PAST 引入外部辅助模式并建立 pattern-missing type 对应
- **vs T1 (ICLR 2026)**：T1 通过 CNN-Transformer 混合处理多变量填补，但未考虑外部信息注入

[^src-past]: [[source-past]]
