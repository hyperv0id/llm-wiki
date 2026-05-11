---
title: "CoFILL"
type: entity
tags:
  - diffusion-models
  - spatio-temporal
  - data-imputation
  - dual-stream-architecture
created: 2026-05-11
last_updated: 2026-05-11
source_count: 2
confidence: high
status: active
---

# CoFILL

**CoFILL** (Conditional Diffusion Model based on Temporal-Frequency Spatiotemporal Imputation) 是由河北工业大学、天津工业大学和南昆士兰大学的研究者于 2025 年提出的时空数据填补框架[^src-cofill]。

## 核心创新

### 1. 非递归扩散结构

传统 RNN/GNN 方法存在误差累积问题——早期预测错误会通过递归结构传播到后续时间步。CoFILL 采用扩散模型的非递归特性，每个时间步的填补独立进行，有效减少误差累积[^src-cofill]。

### 2. 双流特征处理架构

```
输入序列 X
     │
     ├──► TCN (时域卷积) ──┐
     │                     │
     ├──► GCN (图卷积空域) ─┼──► Cross-Attention ──┐
     │                     │                       │
     └──► DCT (频域变换) ───┘                       │
                                                   ▼
                                           条件信息 Ccon
                                                   │
                                                   ▼
                                    噪声预测网络 εθ 逐步去噪
```

- **时域分支**：TCN (Temporal Convolutional Network) + GCN 学习时序依赖和空间关联
- **频域分支**：DCT (Discrete Cosine Transform) 提取周期性和长期趋势
- **融合**：Cross-Attention 机制将两个分支的特征融合为条件信息[^src-cofill]

### 3. 双策略预处理

CoFILL 使用两种预处理策略生成条件输入：
- **Forward Interpolation**：用前一时间步的值填补当前缺失值，保持时空连续性
- **Gaussian Noise**：注入与数据分布一致的高斯噪声，增强数据多样性[^src-cofill]

## 实验结果

| 数据集 | 指标 | 相比 PriSTI 提升 |
|--------|------|-----------------|
| METR-LA (Block) | MAE, MSE | 10.22% |
| PEMS-BAY (Point) | MAE, MSE | ~1% |
| AQI-36 (SF) | MAE, MSE | 3.65% |

消融实验表明，Forward Interpolation 的贡献���大，移除后 MAE 从 8.70 升至 9.15[^src-cofill]。

## 与相关方法的对比

| 方法 | 扩散域 | 时频融合 | 空间建模 |
|------|--------|---------|---------|
| CSDI | 原始域 | 否 | 否 |
| PriSTI | 原始域 | 有限 | 注意力 |
| SpecSTG | 谱域 | 否 | 谱域嵌入 |
| **CoFILL** | 原始域 | **双流 Cross-Attention** | **TCN + GCN** |
| **ImputeFormer** | 原始域 | 投影+嵌入Attention | 隐式（节点嵌入） |
| **GSLI** | 原始域 | 跨特征+跨时间 Transformer | **双尺度学习图**（节点+特征） |

## 局限性

### 计算效率

CoFILL 相比其他扩散填补方法计算开销较大[^src-cofill]：

1. **扩散步数**：T=100（META-LA/PEMS-BAY 为 50），每次填补需要 100 步迭代去噪
2. **双流架构**：时域分支（TCN+GCN）+ 频域分支（DCT）+ Cross-Attention 融合，计算复杂度高
3. **双策略预处理**：Forward Interpolation + Gaussian Noise 两种预填补输入需额外计算

| 方法 | 扩散步数 | 架构复杂度 |
|------|---------|-----------|
| CSDI | 100 | 单流 Transformer |
| PriSTI | ~50 | 时空注意力 |
| **CoFILL** | **100** | **双流 TCN+GCN+DCT+CrossAttn** |

消融实验显示 Forward Interpolation 对性能贡献最大（移除后 MAE 8.70→9.15），但其带来的额外预填补计算也是效率损失的一部分。

## 代码

https://github.com/joyHJL/CoFILL

## 关联页面

- [[diffusion-model]] — 扩散模型理论基础
- [[imputeformer]] — ImputeFormer，低秩引导的 Transformer 时空填补
- [[gsli]] — GSLI，多尺度图结构学习填补（AAAI 2025，处理特征异质性）
- [[spatio-temporal-foundation-model]] — 时空基础模型
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[traffic-forecasting]] — 交通预测

[^src-cofill]: [[source-cofill-spatiotemporal-imputation]]
[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]