---
title: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - transformer
  - multivariate-forecasting
  - ICLR-2024
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: high
status: active
---

# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

**作者**：Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long（清华大学软件学院 + 蚂蚁集团）
**发表**：ICLR 2024 | arXiv:2310.06625v4 (2024-03-14)
**代码**：https://github.com/thuml/iTransformer

## 核心论点

iTransformer 的核心洞察是：**Transformer 在时间序列预测中表现不佳，不是因为 Transformer 组件本身无效，而是因为架构被不当使用。** 传统做法将同一时间步的多个变量嵌入为一个 temporal token，再用注意力建模时间依赖——但同一时间步的变量可能代表不同物理含义、存在系统性时滞、统计分布各异，这种嵌入方式抹除了变量间的独立性并产生无意义的注意力图[^src-itransformer]。

iTransformer 提出将维度反转（invert）：将**每个变量的整条时间序列**嵌入为一个 variate token，然后：

1. **Self-attention** 作用于 variate token 维度，捕获**多变量相关性**（multivariate correlations）
2. **Feed-forward network** 作用于每个 variate token 内部，学习**序列表示**（series representations）
3. **Layer normalization** 作用于每个 variate token，消除不同变量间因不一致度量造成的差异

整个过程不修改 Transformer 的任何原生组件，仅改变它们的应用维度[^src-itransformer]。

## 核心贡献

### 1. 反转 Transformer 组件职责

| 组件 | 传统 Transformer | iTransformer |
|------|------------------|-------------|
| Embedding | 多变量同一时间步 → temporal token | 单变量整条序列 → variate token |
| Self-attention | 建模时间依赖 | 建模多变量相关性 |
| FFN | 作用于 temporal token | 作用于 variate token（学习序列表示） |
| LayerNorm | 归一化多变量表示（融合变量） | 归一化单变量序列表示（消除度量差异） |
| 位置编码 | 需要 | 不需要（序列顺序隐式存储在 FFN 神经元排列中） |

### 2. 框架通用性（iTransformers）

反转框架可直接应用于任何 Transformer 变体（Reformer、Informer、Flowformer、FlashAttention），无需修改组件：

- Transformer + Inverted: 平均 38.9% MSE 提升
- Reformer + Inverted: 36.1% 提升
- Informer + Inverted: 28.5% 提升
- Flowformer + Inverted: 16.8% 提升
- Flashformer + Inverted: 32.2% 提升

高效注意力机制（线性复杂度）可直接作为插件，解决变量数增长时的计算瓶颈[^src-itransformer]。

### 3. 变量泛化能力

由于注意力机制对输入 token 数量灵活，iTransformer 可在 20% 变量上训练、推理时预测全部变量，且性能增幅小于 CI-Transformer（逐变量推理）。FFN 学到的序列表示可在不同变量间迁移——作者解释为 FFN 神经元充当"滤波器"，学习时间序列的内禀属性（振幅、周期性、频谱）[^src-itransformer]。

### 4. 扩展回看窗口

传统 Transformer 随回看窗口增长性能不提升（注意力分散），而 iTransformer 由于 FFN 作用于时间维度，可从扩展的历史信息中获益，与线性预测器的理论期望一致[^src-itransformer]。

## 实验结果

- **7 个真实数据集**（ETT/Exchange/Weather/ECL/Traffic/Solar-Energy/PEMS）+ Market（6 子集）
- 在高维时间序列（Traffic 862 变量、ECL 321 变量）上优势尤为显著
- PEMS 数据集：13/13 首位，Market：28/48 首位
- PatchTST 在 PEMS 波动序列上表现不佳（patch 机制丢失局部关注），iTransformer 整合整条序列表示更鲁棒
- Crossformer（显式跨维度依赖）仍不如 iTransformer——跨变量 patch 交互引入不必要噪声[^src-itransformer]

### 消融实验关键发现

- iTransformer（attention on variate + FFN on temporal）总体最优
- Vanilla Transformer（attention on temporal + FFN on variate）表现最差
- FFN 在时间维度上的作用至关重要——CKA 分析表明 iTransformer 学到更高相似度的表示（低级生成任务偏好高 CKA）
- 注意力图的逐层演化：浅层反映历史变量相关性，深层逐步接近未来变量相关性——说明"编码过去"和"解码未来"在 FFN 前向过程中完成

### 高效训练策略

对高维序列，每 batch 随机采样部分变量训练，推理时预测全部变量。20% 采样率下性能仅轻微下降，内存显著减少。

## 局限与批判

1. **单变量场景退化为堆叠线性预测器**：attention 退化为自连接，时间依赖建模有限
2. **嵌入方式简单**：当前使用 MLP 嵌入整条序列，缺乏更强归纳偏置（如 TCN），对不规则/不等间距序列鲁棒性待验证
3. **注意力复杂度**：变量数 N 大时 $O(N^2)$ 复杂度仍存在，需依赖高效注意力插件
4. **未探索大规模预训练**：作者提出 iTransformer 架构天然支持不同变量数的序列联合训练，为时序基础模型提供方向，但未实验验证

## 与相关工作的定位

论文将 Transformer-based 预测器分为四类（图 3）：
- **(I)** 组件适应（Autoformer, Informer）— 修改注意力
- **(II)** 序列处理（PatchTST, Stationary）— Channel Independence + Patching
- **(III)** 组件+架构同时修改（Crossformer）
- **(IV)** 仅改架构不改组件（**iTransformer**）— 唯一属于此类

iTransformer 与 [[channel-independence]] 正交：CI 将变量完全独立，丢失多变量相关性；iTransformer 保持变量独立嵌入但通过 attention 显式捕获相关性。与 [[crossformer]] 的区别：Crossformer 在跨变量 patch 间做注意力，iTransformer 在变量 token 间做注意力——更简洁且避免时间不对齐 patch 的噪声[^src-itransformer]。

[^src-itransformer]: [[source-itransformer]]
