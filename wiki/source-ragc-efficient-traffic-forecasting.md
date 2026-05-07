---
title: "RAGC: Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution"
type: source-summary
tags:
  - traffic-forecasting
  - graph-neural-network
  - scalability
  - regularization
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# RAGC: Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution

**作者**: Kaiqi Wu, Weiyang Kong, Sen Zhang, Zitong Chen, Yubao Liu（中山大学）  
**时间**: 2026 年 4 月 (arXiv:2506.07179v2)  
**代码**: https://github.com/wkq-wukaiqi/RAGC

## 核心论点

自适应图学习在大规模路网上面临两个根本性问题：(1) 图卷积的 O(N²) 计算复杂度严重限制可扩展性；(2) 占参数主导地位的节点嵌入缺乏正则化，导致过拟合。RAGC 通过三项技术创新协同解决这两个问题。

## 核心贡献

### 1. Efficient Cosine Operator (ECO)
基于余弦相似度的线性复杂度图卷积算子。通过门控机制（softmax + ReLU 逐元素乘积）确保非负性和稀疏性，再对 ℓ2 归一化后的嵌入计算余弦相似度。利用矩阵结合律重排运算顺序，避免显式构建 N×N 邻接矩阵，将复杂度从 O(N²) 降至 O(N)。与 BigST 的随机特征映射近似不同，ECO 不引入近似噪声。

### 2. SSE + 残差差分机制的协同
SSE（随机共享嵌入）通过随机替换节点嵌入注入全局平均噪声进行正则化，但噪声会通过残差连接跨层传播。残差差分机制（$H^{(l)} = H^{(l)}_{mlp} - H^{(l)}_g$）使图卷积权重可以自适应抑制噪声——深层权重趋近单位矩阵时噪声被消除，浅层权重保留噪声以增强正则化。

### 3. 理论分析
推导了 SSE 噪声在扩散图卷积下的传播行为，证明了减法残差使权重矩阵能够以数据驱动方式决定是否抑制全局平均噪声项。

## 实验验证

- **数据集**: LargeST（SD 716、GBA 2,352、GLA 3,834、CA 8,600 节点），12 输入步 → 12 预测步
- **精度**: 四个数据集上所有指标（MAE/RMSE/MAPE）始终最优，较次优 PatchSTG 改进 4.4%–6.4% MAE
- **效率**: 训练速度第 2 快，推理速度第 3 快（仅次于 GSNet 和 BigST，但两者精度较差）
- **消融**: 移除 SSE/RDM/AGC 任一组件均显著降精度；Dropout 和 Laplacian 正则化不如 SSE+RDM

## 局限

- 仅在 LargeST（加州交通流）上验证，未覆盖速度/占用率等指标
- 扩散步 Z 设为 1 避免过平滑，多步扩散的收益未充分探索
- 长视野预测（>12 步）未评估
- SSE 最初为推荐系统设计，其在时空预测中的理论基础尚需更深入分析

## 相关页面

- [[ragc]] — 模型实体页面
- [[efficient-cosine-operator|ECO]] — 线性复杂度图卷积
- [[stochastic-shared-embedding|SSE]] — 随机共享嵌入
- [[residual-difference-mechanism|RDM]] — 残差差分机制
- [[node-embedding-regularization]] — 节点嵌入正则化概念
- [[traffic-forecasting]] — 交通预测方法概览
- [[large-scale-spatial-temporal-graph]] — 大规模时空图预测

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
