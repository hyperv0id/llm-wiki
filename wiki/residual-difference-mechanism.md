---
title: "Residual Difference Mechanism (RDM)"
type: technique
tags:
  - residual-learning
  - noise-suppression
  - graph-convolution
  - traffic-forecasting
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# Residual Difference Mechanism (RDM)

RDM 是 RAGC 中的核心噪声抑制机制，通过**减法残差**而非传统加法残差，抑制 [[stochastic-shared-embedding|SSE]] 引入的全局平均噪声在深层网络中的传播[^src-ragc-efficient-traffic-forecasting]。

## 动机

### 传统残差连接的噪声传播问题

在 L 层残差网络中：
$$H^{(l)} = H^{(l-1)} + F^{(l)}(H^{(l-1)})$$

最终输出 $H^{(L)} = \tilde{e}_i + \sum_{l=1}^{L} F^{(l)}(H^{(l)})$，其中 SSE 扰动嵌入 $\tilde{e}_i$ 通过残差路径直接传播到输出层，噪声持续影响所有层[^src-ragc-efficient-traffic-forecasting]。

### 图卷积的噪声过滤能力

理论分析表明，扩散图卷积输出的期望为：
$$\mathbb{E}[h_i] = p\bar{e}\sum_{z=0}^{Z} W^{(z)}_g + (1-p)\sum_{z=0}^{Z}\sum_{k=1}^{N} A^{(z)}_{ik} e_k W^{(z)}_g$$

由于 A 行归一化（$\sum_k A^{(z)}_{ik} = 1$），噪声项 $p\bar{e}$ 可通过权重矩阵 $W_g$ 抑制[^src-ragc-efficient-traffic-forecasting]。

差分期望：
$$\mathbb{E}[\tilde{e}_i - h_i] = p\bar{e}(I - \sum_{z=0}^{Z} W^{(z)}_g) + (1-p)(e_i - \sum_{z=0}^{Z}\sum_{k=1}^{N} A^{(z)}_{ik} e_k W^{(z)}_g)$$

当权重和 $\tilde{W}_g \approx I$ 时，全局平均噪声项 $p\bar{e}$ 被有效消除。

## 实现

$$H^{(l)} = H^{(l)}_{mlp} - H^{(l)}_g$$

其中 $H^{(l)}_{mlp}$ 是 MLP 输出，$H^{(l)}_g$ 是图卷积输出（全局趋势）[^src-ragc-efficient-traffic-forecasting]。

### 信号分解视角

- $H^{(l)}_g$：平滑表示，反映节点邻居间的全局趋势
- $H^{(l)}$：残差，捕获节点级波动

两者共同构成每个节点的交通流模式。全局趋势通过跳跃连接 $H_{skip} = \sum_{l=1}^{L} H^{(l)}_g$ 保留，在回归层与残差分支求和输出。

## 为什么减法优于加法/拼接

消融实验验证[^src-ragc-efficient-traffic-forecasting]：

| 方式 | SD MAE | CA MAE |
|------|--------|--------|
| RDM（减法） | 16.16 | 16.40 |
| 加法 | 16.35 | 16.54 |
| 拼接 | 16.38 | 16.71 |

减法融合最优，因为其数学性质使权重矩阵 $W_g$ 能自适应地决定是否抑制噪声——当 $\tilde{W}_g \approx I$ 时噪声被消除，否则保留以增强正则化效果。

## 自适应噪声阻断的可视化证据

权重矩阵 $\tilde{W}_g$ 的可视化显示[^src-ragc-efficient-traffic-forecasting]：
- **浅层**：$\tilde{W}_g$ 不接近单位矩阵 → 模型有意保留全局平均噪声以增强正则化
- **中/深层**：$\tilde{W}_g$ 逐步趋近单位矩阵 → 模型自适应抑制噪声，防止其对最终预测产生负面影响

## 相关页面

- [[ragc]] — RDM 所在的完整模型
- [[stochastic-shared-embedding|SSE]] — RDM 抑制的噪声来源
- [[efficient-cosine-operator|ECO]] — RDM 中的图卷积算子
- [[node-embedding-regularization]] — 嵌入正则化概念

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
