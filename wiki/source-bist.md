---
title: "BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction"
type: source-summary
tags:
  - spatiotemporal
  - traffic-forecasting
  - mlp
  - lightweight
  - bidirectional
  - gmf
created: 2026-06-10
last_updated: 2026-06-10
source_count: 0
confidence: high
status: active
---

# BiST — Source Summary

**Authors**: Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, Yang Wang (University of Science and Technology of China)
**Venue**: PVLDB 18(6): 1663–1676, 2025
**Code**: <https://github.com/PoorOtterBob/BiST>

## 核心贡献

BiST 针对时空预测中两个被忽视的问题：(1) **输入-标签时空偏差**——现有模型假设输入数据和标签的时空相关性一致，但实际上相似的输入可能对应迥异的标签（反之亦然），导致预测误差；(2) **高昂的计算复杂度**——Transformer 类时空模型的时间和显存开销随节点数二次增长，难以扩展到大规模数据。

BiST 的主要贡献：

1. **时空动力学理论**：基于 Gaussian Markov Random Field (GMRF) 推导，证明引入标签信息后，最优预测 = 前向基预测 + 校正项。校正项包含扩散核和残差，从理论上建立了双向学习范式。
2. **双向架构**：前向过程用纯 MLP 捕获输入时空相关性生成基预测；后向过程建模输入-标签的残差偏差，生成校正项修正基预测。
3. **时空残差解耦模块**：将时空特征分解为节点共享的上下文特征（通过虚拟聚类学习）和节点个性化特征，有效捕获标签与输入之间的不一致信息。
4. **自适应残差扩散**：通过可学习的扩散核对残差进行平滑传播。

## 实验结果

- 13 个数据集（含 16,972 节点的 XTraffic 和跨度 20 年的 XXLTraffic）
- 对比 26 个基线模型
- 相对 SOTA 提升 8.13%，但训练时间仅为 SOTA 的 1.86%，显存占用的 7.36%
- 在数据突增/骤降场景下表现显著优于现有模型，验证了处理时空偏差的能力

## 局限

- 主要验证于交通和空气质量领域，其他时空预测任务（如天气、电力）的泛化性有待检验
- 纯 MLP 架构在极长期预测上的能力边界未充分探索
