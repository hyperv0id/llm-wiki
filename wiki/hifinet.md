---
title: "HiFiNet"
type: entity
tags:
  - graph-neural-network
  - road-network
  - frequency-decomposition
  - hierarchical-modeling
  - spectral-methods
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# HiFiNet

**HiFiNet**（Hierarchical Frequency-Decomposition Network）是一个面向道路网络表示学习的统一空间-频谱 GNN 框架，由 Ma、Wang（北航）和 U（澳门大学）提出，发表于 AAAI 2026[^src-hifinet]。

## 核心思想

现有图神经网络建模道路网络时存在**空间-频谱失配**（spatial-spectral misalignment）：空间方法捕获局部拓扑但趋于过平滑（低通滤波），频谱方法分析全局频率但忽略局部变化[^src-hifinet]。HiFiNet 通过三层层次结构（路段 → 街区 → 区域）和频率分解模块统一了空间与频谱建模。

## 架构

HiFiNet 由两个关键组件构成[^src-hifinet]：

### 层次结构建模（Hierarchical Architecture）

1. **路段上下文嵌入**：路段 ID、车道数、长度、地理坐标 → 拼接 + FFN 编码
2. **街区图构建**：跨注意力 softmax 计算路段→街区软分配矩阵 $A_{SL}$，汇聚形成街区特征和邻接矩阵
3. **区域图构建**：同理形成街区→区域分配 $A_{LR}$，构建三层层次图 $H = \\langle S \\cup L \\cup R, \\mathcal{E}\\rangle$
4. **低频特征自顶向下传播**：区域 GAT → 区域→街区反池化(Unpooling) → 街区 GAT → 街区→路段反池化 → 路段 GAT

### 频率分解建模（Frequency Decomposition）

- **分解阶段**：高频特征 $H_S^h = H_S - H_S^l$（原始特征减去低频特征）
- **更新阶段**：低频和高频分别通过 TGT 更新
- **重构阶段**：$\\hat{H}_S = \\beta \\cdot \\tilde{H}_S^l + (1-\\beta) \\cdot \\tilde{H}_S^h$

### 拓扑感知图 Transformer（TGT）

注意力权重融合全局自注意力和局部拓扑[^src-hifinet]：

$$\\text{ATT} = \\alpha \\cdot \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right) + (1-\\alpha) \\cdot A_S$$

其中 $\\alpha$ 为可学习参数，平衡全局与局部信息。

## 理论基础

**定理 1**：满足等分割和行归一化的层次分配矩阵近似执行**频谱低通滤波**——保留平滑信号分量，衰减高频噪声[^src-hifinet]。这解释了为何层次反池化传播后得到的 $H_S^l$ 是低频特征，并可自然地通过减法提取高频分量。

## 实验结果

在 Beijing、Chengdu、Xi'an 三个真实数据集上，面向四个下游任务全面 SOTA[^src-hifinet]：
- 下一位置预测（Next Location Prediction）
- 路段标签分类（Label Classification）
- 目的地预测（Destination Prediction）
- 路径规划（Route Planning）

消融实验证明层次结构（NB < NL < NR < full）和频率分解（NLF < NHF < full）均至关重要。

## 与其他方法的关系

HiFiNet 针对的是**道路网络表示学习**（road network representation learning），这与[[traffic-forecasting|交通预测]]中直接预测速度/流量的任务不同——HiFiNet 输出的是路段嵌入（representations），适用于多种下游任务[^src-hifinet]。但其层次图建模和频谱分析方法可延伸至更广泛的时空图域。

与 [[stgcn|STGCN]] 的对比：STGCN 使用预定义图结构 + 纯卷积，是预测任务；HiFiNet 使用可学习层次分配 + 频率分解，是表示学习任务。两者共享"将城市路网建模为图"的范式[^src-hifinet]。

## 关联页面

- [[source-hifinet]] — 源文件摘要
- [[road-network-representation-learning]] — 道路网络表示学习概念
- [[graph-frequency-decomposition]] — 图频率分解
- [[over-smoothing-in-gnns]] — GNN 过平滑问题
- [[topology-aware-graph-transformer]] — TGT 技术
- [[traffic-forecasting]] — 交通预测总览
- [[stgcn]] — STGCN，谱域图卷积交通预测

[^src-hifinet]: [[source-hifinet]]
