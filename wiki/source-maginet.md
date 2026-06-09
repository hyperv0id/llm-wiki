---
title: "MagiNet: Mask-Aware Graph Imputation Network for Incomplete Traffic Data"
type: source-summary
tags:
  - data-imputation
  - traffic-forecasting
  - graph-neural-network
  - spatio-temporal
  - attention
  - arxiv-2024
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# MagiNet: Mask-Aware Graph Imputation Network for Incomplete Traffic Data

**作者**：Jianping Zhou, Bin Lu, Zhanyu Liu, Siyu Pan, Xuejun Feng, Hua Wei, Guanjie Zheng, Xinbing Wang, Chenghu Zhou（上海交通大学、亚利桑那州立大学、中国科学院）。
**出处**：arXiv 2406.03511（2024 年 6 月），后发表于 ACM TKDD（2025）[^src-maginet]。

## 核心问题

交通数据采集中，探测器故障与通信失败导致缺失值普遍存在；填补缺失值对智能交通系统（ITS）的数据分析和决策至关重要[^src-maginet]。论文指出现有深度填补方法的两个关键缺陷：(1) 普遍采用**零预填充**（zero pre-filling）初始化缺失值并用掩码矩阵记录其位置，但预填充不可避免地引入噪声、误导特征学习；(2) 现有时空填补方法（[[grin]]、GA-GAN）在预填充数据上捕获时空相关性，忽略内在动态变化，导致**过平滑插值**（over-smoothing），尤其在连续/动态缺失位置失真[^src-maginet]。作者用 Seattle 数据集实证：带预填充的填补性能显著劣于不带预填充[^src-maginet]。

## 方法

MagiNet 是编码器-解码器框架[^src-maginet]：

- **自适应掩码时空编码器（AMSTenc）**：将不完整数据分解为特征矩阵 X、掩码矩阵 M、缺失矩阵 Z。观测段经观测嵌入层得 X_o，缺失段经**可学习掩码嵌入层**得 Z_u，按掩码组合 X_p = X_o ⊙ M + Z_u ⊙ (1−M)，再加可学习时序位置嵌入。由此不靠零初始化即可自适应表示缺失值，避免引入噪声[^src-maginet]。
- **掩码感知时空解码器（MASTdec）**：堆叠多个时空块。每块含**掩码感知时空注意力（MASTatt）**——多头自注意力计算时间注意力分数，跨块用注意力残差连接累加，并将掩码 M 乘入注意力以屏蔽缺失值对观测的影响（C = Softmax(M ⊙ A^(l))V）；随后计算掩码感知空间注意力 S_att[^src-maginet]。
- **基于注意力的时空聚合**：用 Chebyshev 多项式图卷积聚合空间信息，并将 S_att 作为权重注入图卷积核（T_k(L̃) ⊙ S_att），动态调整不完整数据下的信息聚合；再用多尺度门控时间卷积（不同核 K=3,5,7）传播观测时间点信息到缺失时间点[^src-maginet]。
- **投影层与训练**：拼接各块输出经两层 FC 得填补结果，仅在缺失位置用 L1 损失训练，Adam 优化[^src-maginet]。

## 实验与结果

在 METR-LA、Seattle、Chengdu、Shenzhen、PEMS-BAY 五个真实交通数据集上，MCAR 50% 缺失率下评估，对比统计/机器学习方法、5 个交通预测模型（预填充用于填补）和 8 个填补基线（[[grin]]、[[pristi|PriSTI]]、GA-GAN、BRITS 等）[^src-maginet]。MagiNet 平均 RMSE 提升 4.31%、MAPE 提升 3.72%；相比预填充的交通预测方法平均提升 7.56%/8.87%[^src-maginet]。消融显示 zero/mean prefill 与 w/o AMSTenc 均劣于完整模型，验证可学习缺失编码的价值；移除 MASTdec 损失最大[^src-maginet]。

## 局限

仅在 PEMS-BAY（低方差数据集）上 [[pristi|PriSTI]] 略优于 MagiNet，作者归因于扩散方法的多步生成更适合低方差数据[^src-maginet]。实验聚焦 MCAR 缺失模式；作者将概率化填补和更大规模数据集的可扩展性列为未来工作[^src-maginet]。

[^src-maginet]: [[source-maginet]]
