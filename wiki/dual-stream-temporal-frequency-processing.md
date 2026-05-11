---
title: "Dual-Stream Temporal-Frequency Processing"
type: technique
tags:
  - spatio-temporal
  - feature-extraction
  - time-frequency
  - dual-stream
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: high
status: active
---

# 双流时频处理 (Dual-Stream Temporal-Frequency Processing)

双流时频处理是 CoFILL 框架中用于同时捕捉时序数据中短期变化和长期周期性模式的技术[^src-cofill]。

## 核心思想

时空数据同时包含：
- **短期变化**：瞬时波动、突变、局部模式（时域擅长捕捉）
- **长期趋势**：周期性、季节性、渐进变化（频域擅长捕捉）

传统方法往往只关注其中一个方面，导致信息丢失。双流架构并行处理两个域，然后通过注意力机制融合[^src-cofill]。

## 架构

```
输入 X ∈ R^{N × L} (节点数 × 时间步)
     │
     ├──► 时域分支 ──► TCN + GCN ──┐
     │                              │
     └──► 频域分支 ──► DCT ─────────┼──► Cross-Attention ──► Ccon
                                    │
                      (可学习的 Q, K, V 投影)
```

### 时域分支

1. **TCN** (Temporal Convolutional Network)：使用门控因果卷积
   - $H_{out} = P \odot \sigma(Q)$
   - $\odot$ 为 Hadamard 积，$\sigma$ 为 sigmoid 门控
   - 捕捉局部时间依赖[^src-cofill]

2. **GCN** (Graph Convolutional Network)：学习空间依赖
   - $H_{spatial} = \sigma(AGCN \cdot H_{temporal} \cdot W)$
   - $AGCN = D^{-1/2}(A+I)D^{-1/2}$ 为归一化邻接矩阵
   - 捕捉节点间的空间关联[^src-cofill]

### 频域分支

**DCT** (Discrete Cosine Transform)：将时域信号变换到频域

$$\hat{H}[m] = \sum_{t=0}^{T-1} H[t] \cdot \cos\left(\frac{\pi}{T}(t + \frac{1}{2})m\right)$$

- 低频系数 ($m \to 0$)：捕获长期稳定趋势
- 高频系数 ($m \to T$)：捕获周期性模式[^src-cofill]

## 融合：Cross-Attention

$$Q = H_{temporal} \cdot W_Q$$
$$K = \hat{H}_{frequency} \cdot W_K$$
$$V = \hat{H}_{frequency} \cdot W_V$$

$$C_{con} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

其中：
- $Q$ 来自时域特征（保留局部变化信息）
- $K, V$ 来自频域特征（编码全局周期信息）
- 融合后得到同时包含局部细节和全局模式的丰富表示[^src-cofill]

## 在 CoFILL 中的应用

1. **条件信息模块**：提取的特征用于引导噪声预测网络
2. **双策略预处理**：Forward Interpolation + Gaussian Noise 两种预填补输入
3. **噪声预测网络**：Temporal Attention + Spatial Attention 级联结构

## 关联技术

- [[timesnet]] — TimesNet 也使用 2D-variation 将时序转为"图像"处理
- [[frequency-enhanced-attention]] — FEDformer 的频域注意力机制
- [[spectral-recurrent-encoder]] — SpecSTG 的谱域循环编码器
- [[adaptive-frequency-fusion]] — UniCA 的自适应频域融合

## 适用场景

- 多尺度时序模式挖掘
- 同时存在短期波动和长期周期性的数据
- 需要同时利用时间依赖和空间关联的时空图数据

[^src-cofill]: [[source-cofill-spatiotemporal-imputation]]