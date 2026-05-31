---
title: "xCPD: Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition"
type: source-summary
tags:
  - time-series
  - forecasting
  - graph-spectral
  - channel-dependency
  - mixture-of-experts
  - plugin
  - ICLR-2026
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# xCPD: Routing Channel-Patch Dependencies with Graph Spectral Decomposition

**Authors**: Dongyuan Li (UTokyo), Shun Zheng, Chang Xu, Jiang Bian (MSRA), Renhe Jiang (UTokyo)

**Venue**: ICLR 2026 | **Code**: [github.com/Clearloveyuan/xCPD](https://github.com/Clearloveyuan/xCPD)

## 核心贡献

xCPD 是一个通用的、模型无关的轻量级 plugin，通过图频谱分解（graph spectral decomposition）在 patch 级别自适应路由通道依赖。"x" 指频谱分解，CPD 指 Channel-Patch Dependencies [^src-xcpd]。

xCPD 直接将建模单元从整个 channel 下沉到 **channel-patch**（通道内的局部时间段），将每个 channel-patch 视为图中的一个节点，在频谱域中显式区分低频（平滑趋势）、中频（局部波动）和高频（突变/噪声）依赖，通过动态 Mixture-of-Experts 路由机制为每个 patch 自适应选择频率专属滤波器 [^src-xcpd]。

## 三个核心模块

### (A) Spectral Channel-Patch Embedding
- 将 backbone 输出按时间维切分为非重叠 patch（每个 patch → 一个图节点），展平后得到 $n = C \times N$ 个节点 [^src-xcpd]
- 用余弦相似度构建稠密邻接矩阵 $A_t$，学习**共享图傅里叶基 (shared graph Fourier basis)** $U$，确保跨 batch 的频谱域一致性，其合法性由 Davis-Kahan 定理保证（定理 4.1）[^src-xcpd]
- 通过 $X^{\text{spc}} = U^\top X^{\text{emb}}$ 将节点嵌入投影到频谱域 [^src-xcpd]

### (B) Spectral Channel-Patch Grouping
- 使用可学习边界 $\tau_1, \tau_2$ 将频段自适应划分为低频/中频/高频三个波段 [^src-xcpd]
- 定理 4.2 定义**频谱能量响应** $S_{i,j} = \|U_{i,j} \cdot X^{\text{spc}}_{j,:}\|^2$，量化每个节点对不同频率的响应强度，且保证能量在空域和频谱域之间等价保真 [^src-xcpd]
- 通过 softmax 分类将节点分配到主导频率组，并在每个 ego-graph 内构建频率感知的子图 [^src-xcpd]

### (C) Spectral Channel-Patch Routing with DyMoE
- 设计三个专属滤波器（低/中/高频），分别构造强调对应频谱成分的邻接矩阵 [^src-xcpd]
- **Dynamic MoE (DyMoE)**：与传统 Top-K 固定选择不同，DyMoE 根据累计概率阈值 $\tau$ 自适应选择 1-3 个 expert，路由分数由确定性分量 + 噪声分量组成 [^src-xcpd]
- 在每个 ego-graph 上通过 GNN 消息传递学习通道-patch 间隐藏关系 [^src-xcpd]
- 最终通过**门控双路径残差校正**合并 GNN 路径（捕获跨变量频谱依赖）和 Linear 路径（保留 CI 细化），当门控值趋近零时退化为 backbone 预测 [^src-xcpd]

训练目标：MSE loss + entropy loss（防 expert 低置信度）+ balance loss（防 expert 负载不均）[^src-xcpd]。

## 关键实验结果

**长期预测**：在 9 个数据集 (144 实验设置) 上，xCPD 在 4 个 backbone（DLinear/PatchTST/TSMixer/TimesNet）上一致提升。在高维数据集上增益最大——Electricity (321 变量) 平均 MSE 降低 4%-7%，Traffic (862 变量) 降低 4%-7% [^src-xcpd]。改进幅度与数据集的频谱复杂度正相关：富频谱变动的数据集（Electricity/Traffic）获益最大，规律性强的数据集（ETT/Solar）获益 1%-3% [^src-xcpd]。

**对比 CCM**：CCM 增加 60-100% 显存和 4-5× 训练时间（迭代聚类），xCPD 显存恒定 ~7GB、每 iter <10ms，效率优势显著 [^src-xcpd]。对比 PCD（Transformer 限定），xCPD 在 PatchTST 上全面优于 PCD [^src-xcpd]。

**短期预测**：M4 数据集（SMAPE/MASE/OWA）和 Stock 数据集上一致提升，长期预测受益更显著（季节性模式的频率分解），短期预测通过 patch 级去噪维持增益 [^src-xcpd]。

**零样本迁移**：48 个 zero-shot 设置中一致提升，CI 模型获益更大（DLinear/PatchTST 分别 12.0%/15.2% vs TSMixer/TimesNet 6.7%/11.1%）[^src-xcpd]。

**消融**：移除共享傅里叶基性能下降最大；移除频率分区影响较小（三波段分解本身提供强归纳偏置）；移除节点分组或滤波器则显著下降 [^src-xcpd]。

**非平稳鲁棒性**：在施加 ±12 时间步偏移和幅度缩放扰动后，xCPD 仅边缘退化；移除共享基则 MSE 增 8.3% [^src-xcpd]。

## 局限性

- 依赖图频谱分解，通道数/patches 数极大时引入中等开销 [^src-xcpd]
- 假设频谱能量可有效引导依赖选择，在高度不规则或频率分布剧烈跨域漂移时可能不成立 [^src-xcpd]
- 集成到复杂 backbone（如自回归/层次化模型）可能需要额外调参 [^src-xcpd]

[^src-xcpd]: [[source-xcpd]]
