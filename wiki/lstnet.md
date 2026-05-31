---
title: "LSTNet"
type: entity
tags:
  - time-series
  - multivariate
  - deep-learning
  - cross-dimension
  - cnn
  - rnn
  - forecasting
  - SIGIR-2018
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# LSTNet

LSTNet (Long- and Short-term Time-series Network) 是首个显式建模多变量时间序列 (MTS) 中**跨维度依赖**和**多尺度时间模式**的深度学习框架，由 Lai, Chang, Yang 和 Liu (CMU) 在 SIGIR 2018 提出 [^src-lstnet]。截至 2026 年，被引用约 1,728 次，是 MTS 深度学习路线的开山之作。

## 问题设定

给定 n 个变量的 T 步历史观测，预测未来 h 步。现实 MTS 数据（如交通流量、电力消耗）同时包含短期局部模式（早晚高峰）和长期周期性模式（工作日 vs 周末），传统 ARIMA/VAR 模型无法区分和显式建模这两类模式的交互 [^src-lstnet]。

## 架构

LSTNet 由四个模块组成，最终预测为非线性部分与线性部分的逐元素和 [^src-lstnet]：

```
输入 X ∈ R^(n×T)
    │
    ├──► CNN（跨维度局部模式）──► GRU（长期依赖）──┐
    │                        └──► Skip-RNN（超长周期）──┤
    │                                                  ├──► 全连接 ──► Ŷ^D
    │                                                  │                +
    └──► AR 线性模型 ──────────────────────────────────┘          Ŷ = Ŷ^D + Ŷ^L
```

### 1. 卷积层 (CNN)

无池化的卷积层，滤波器尺寸为 ω × n（高度 = 变量数），在时间轴上滑动。每个滤波器提取一个跨所有变量的局部模式——这是 MTS 领域首次用 CNN 显式建模**跨维度依赖** [^src-lstnet]。输出尺寸 d_c × T。

### 2. 循环层 (RNN)

标准 GRU，但使用 RELU 替代 tanh 作为隐藏更新激活函数。作者实验发现 RELU 的梯度回传更稳定 [^src-lstnet]。

### 3. 跳跃循环层 (Skip-RNN)

这是 LSTNet 最关键的创新。标准 GRU/LSTM 因梯度消失难以捕获极长周期依赖（如 24 小时前的同一时刻）。Skip-RNN 在相邻周期的同相位隐藏状态之间添加跳跃连接：r_t 和 u_t 的门控输入来自 h_{t-p}（而非 h_{t-1}），其中 p 是周期长度 [^src-lstnet]。

对于小时级交通和电力数据，p=24；对于太阳能数据（10 分钟采样），需调参。当 p 未知或周期动态变化时，可用 **LSTNet-Attn**（时间注意力）替代。

### 4. 自回归组件 (AR)

这是一个关键但容易被忽视的设计。神经网络模型对输入尺度变化不敏感——当电力消耗因节假日突然飙升时，纯网络无法及时响应。并行运行的经典 AR 线性模型使最终输出对输入尺度保持敏感 [^src-lstnet]。

消融实验中，移除 AR 组件在所有数据集上造成**最大的性能下降**，证明了其必要性 [^src-lstnet]。合成实验进一步验证：仅在训练集注入尺度变化后，LSTNet 能适应测试集的尺度漂移，而纯 GRU 完全失败 [^src-lstnet]。

### LSTNet-Attn 变体

当周期 p 未知时，LSTNet-Attn 用注意力机制替代 Skip-RNN：计算当前隐藏状态与过去 q 个窗口位置的注意力权重，加权汇总上下文向量 [^src-lstnet]。在无周期性的 Exchange-Rate 数据集上，LSTNet-Attn 表现优于 LSTNet-skip。

## 性能

在 Traffic、Solar-Energy、Electricity、Exchange-Rate 四个数据集上，horizon = {3, 6, 12, 24}，LSTNet-skip 获得 **17 项最佳**，LSTNet-Attn 获得 **7 项**[^src-lstnet]：

| 数据集 | 相对 RNN-GRU 提升 (horizon=24, RSE) | 周期性 |
|--------|--------------------------------------|--------|
| Solar-Energy | 9.2% | 有 |
| Traffic | 11.7% | 有（日+周） |
| Electricity | 22.2% | 有（日+周） |
| Exchange-Rate | 略差于 AR/LRidge | 无 |

## 在跨维度依赖建模中的历史位置

LSTNet 是 [[cross-dimension-dependency|跨维度依赖]] 建模路线的起点 [^src-lstnet]：

| 时间 | 模型 | 跨维度方法 |
|------|------|-----------|
| 2018 | **LSTNet** | CNN 提取变量间局部模式 |
| 2020 | [[mtgnn|MTGNN]] | 自适应图学习 + GNN |
| 2023 | [[crossformer|Crossformer]] | 2D embedding + 两阶段注意力 |

LSTNet 之后，MTGNN (KDD 2020) 将跨维度依赖推广到图神经网络范式，Crossformer (ICLR 2023) 将其引入 Transformer。三者共同构成跨维度依赖建模的三条技术路线 [^src-lstnet]。

与 [[channel-independence|Channel Independence (CI)]] 策略对比：LSTNet 是 CD（跨维度依赖）路线的早期代表，主动利用变量间关系；CI 路线（如 PatchTST）完全忽略变量间关系。LSTNet 的实验间接支持 CI 的合理性——在 Exchange-Rate（弱变量相关性）上，AR/LRidge（相当于独立模型）反超 LSTNet [^src-lstnet]。

## 意义

LSTNet 的核心洞见——将线性模型与非线性神经网络**并行组合**（而非串行级联）——在后续工作中被反复验证。其 Skip-RNN 思想启发了周期性建模的一系列工作（如 [[cyclenet|CycleNet]] 的 RCF），AR 组件的尺度敏感性洞察至今仍是 MTS 预测的重要教训。

[^src-lstnet]: [[source-lstnet]]
