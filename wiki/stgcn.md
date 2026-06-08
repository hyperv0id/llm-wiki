---
title: "STGCN"
type: technique
tags:
  - traffic-forecasting
  - graph-neural-network
  - spatial-temporal
  - convolutional-neural-network
  - spectral-methods
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# STGCN

**STGCN**（Spatio-Temporal Graph Convolutional Networks）是首个**纯卷积时空图网络**，由 Yu, Yin & Zhu（北京大学）于 IJCAI 2018 提出[^src-stgcn]。核心创新：用谱域图卷积建模空间依赖，用门控 1D 因果卷积替代 RNN 建模时间动态，通过"时间→空间→时间"三明治架构（ST-Conv Block）将时空两维度统一在纯卷积框架内，实现比 RNN 系方法快一个数量级的训练速度[^src-stgcn]。

## 架构

### ST-Conv Block

STGCN 的核心构建块，由三层组成[^src-stgcn]：

```
输入 v^l (3D: M × n × C^i)
  │
  ▼ 时间门控卷积 (上层)
  │  Kt=3, GLU, 输出 C1
  │  残差连接
  ▼
空间图卷积 (中层)
  │  图卷积核 Θ, 输出 C2
  │  ReLU + Layer Normalization
  │  瓶颈策略: C2 << C1
  ▼
时间门控卷积 (下层)
  │  Kt=3, GLU, 输出 C1
  │  残差连接
  ▼
输出 v^{l+1}
```

完整前向公式[^src-stgcn]：

$$v^{l+1} = \Gamma_1^l *_T \text{ReLU}\left(\Theta^l *_G \left(\Gamma_0^l *_T v^l\right)\right)$$

两个 ST-Conv Block 堆叠意味着数据在"时间→空间→时间→时间→空间→时间"的路径上走两遍，时空依赖被多层次地熔铸[^src-stgcn]。

### 图卷积（空间建模）

谱域图卷积算子[^src-stgcn]：

$$\Theta *_G x = U \Theta(\Lambda) U^T x$$

两种近似变体：

| 变体 | 公式 | 复杂度 | 特点 |
|------|------|--------|------|
| STGCN(Cheb) | $\sum_{k=0}^{K-1} \theta_k T_k(\tilde{\Lambda})$，$K=3$ | $O(K\|E\|)$ | 显式参数化 3 跳感受野 |
| STGCN(1st) | $\theta(\tilde{D}^{-1/2}\tilde{W}\tilde{D}^{-1/2})x$ | $O(\|E\|)$ | 单参数/层，层叠实现多跳 |

一阶近似通过堆叠 K 层等价于 K 跳感受野，且层间有 ReLU 非线性——比多项式近似的一次性计算更有表达力[^src-stgcn]。

### 门控时间卷积（时间建模）

时间卷积核 $\Gamma \in \mathbb{R}^{K_t \times C_i \times 2C_o}$，$K_t=3$，因果（无 padding）。门控线性单元[^src-stgcn]：

$$\Gamma *_T Y = P \odot \sigma(Q)$$

$P$ 和 $Q$ 是卷积输出沿通道维度的等分。$\sigma(Q)$ 是 sigmoid 门控——比 LSTM 的三重门（遗忘/输入/输出）轻一个数量级。因果卷积保证不偷看未来，GLU 门控控制信息流，全时间步可并行[^src-stgcn]。

### 瓶颈策略

中间图卷积的通道数被大幅压缩（如 64→16→64），因为图卷积的空间复杂度 $O(K|E|C_i C_o)$ 与通道数二次相关。压缩到 1/4 等于图卷积计算量降为 1/16[^src-stgcn]。

### 输出与损失

最后一个 Block 后接单层时间卷积 → 全连接线性变换 → 标量速度预测。损失函数为朴素 L2[^src-stgcn]：

$$L(\hat{v}; W_\theta) = \sum_t \|\hat{v}(v_{t-M+1},...,v_t, W_\theta) - v_{t+1}\|^2$$

## 实验性能

### 数据集与设置

| 数据集 | 节点数 | 图构建 | 输入 | 预测 |
|--------|--------|--------|------|------|
| BJER4 | 12 条路 | 传感器部署图（有向） | 60min（12 步） | 15/30/45min |
| PeMSD7(M) | 228 传感器 | 高斯核距离（无向） | 同上 | 同上 |
| PeMSD7(L) | 1026 传感器 | 同上 | 同上 | 同上 |

### 主结果

STGCN(Cheb) 在三数据集、三预测长度、三指标（MAE/MAPE/RMSE）上**全面 SOTA**[^src-stgcn]。关键发现：
- 图卷积系方法（STGCN, GCGRU）一致优于非图方法（FC-LSTM, FNN），验证了"图优于网格"[^src-stgcn]
- STGCN(1st) 在小数据集上精度略逊，大数据集上更快（PeMSD7(L) 上比 Chebyshev 快约 20%），精度损失可控[^src-stgcn]
- 早晚高峰的关键拐点（如拥堵转畅通）上，STGCN 比 RNN 系方法更早做出反应——说明图卷积让模型对"交通波的图扩散"更敏感[^src-stgcn]

### 训练效率

| 模型 | 训练时间 | 参数量 | vs STGCN |
|------|---------|--------|----------|
| STGCN(Cheb) | **272s** | 4.54×10⁵ | — |
| STGCN(1st) | 271s | 相似 | — |
| GCGRU | 3825s | ≈6.7×10⁵ | **14× 慢** |
| FC-LSTM | 未报告 | >2×10⁷ | — |

PeMSD7(L) 上 STGCN 训练 1554s vs GCGRU 19511s（**12.5 倍加速**）。加速的核心原因不是计算量减少，而是卷积天然支持 GPU 全并行——RNN 的串行依赖让它只能用 GPU 并行能力的零头[^src-stgcn]。

## 设计哲学

### 为什么"三明治"而非其他排列？

论文尝试过其他排列（空间-时间-空间等），三明治结构（时间-空间-时间）最优[^src-stgcn]。原因：
1. 两侧时间卷积负责通道压缩/扩展 → 中间图卷积的计算负担最小
2. 时间→空间的信号变换让图卷积接收的是时间变换后的特征，而非原始输入
3. 残差从输入直通输出 → 梯度可跳过中间层直接回传

### 为什么 GLU 而非普通 ReLU？

门控机制让网络学会"选择性通过"信息——不是所有时间步的所有通道都对空间传播有用。论文未直接消融 GLU vs ReLU，但后续工作（[[diffstg|DiffSTG]] 的 UGnet）继承了 GLU，间接验证其有效性[^src-stgcn]。

### 为什么一阶近似在大图上更快？

一阶近似的图卷积是简单的稀疏矩阵乘法 $\tilde{D}^{-1/2}\tilde{W}\tilde{D}^{-1/2}x$。Chebyshev 多项式 $K=3$ 需要递归计算 $T_k$，每个递归步骤都有矩阵乘法。在大稀疏图上，递归的多步矩阵乘法比直接一阶叠层更慢[^src-stgcn]。

## 局限与后续演进

| 局限 | 解决方案 | 提出时间 |
|------|---------|---------|
| 邻接矩阵预定义 | [[gwnet|GWNet]] 自适应图学习 | IJCAI 2019 |
| 单数据集专用 | UrbanDiT 基础模型 | NeurIPS 2025 |
| 确定点估计 | DiffSTG/SpecSTG 概率预测 | AAAI 2023 / 2024 |
| 图结构时不变 | ASTGCN 时空注意力 | AAAI 2019 |
| 仅速度单变量 | 后续多维联合预测 | — |

STGCN 与同年发表的 [[dcrnn|DCRNN]]（ICLR 2018）共同确立了时空图建模的两条技术路线：DCRNN 使用扩散卷积 + DCGRU（RNN 系），STGCN 使用谱图卷积 + 门控因果卷积（纯 CNN 系）[^src-stgcn]。STGCN 的"纯卷积 + 预定义图"范式历经 GWNet、ASTGCN、DiffSTG、SpecSTG 的持续扩展，最终汇入 UrbanDiT 的通用时空基础模型路线。

## 关联页面

- [[source-stgcn]] — 源文件摘要
- [[traffic-forecasting]] — 交通预测总览
- [[gated-linear-units]] — GLU 门控线性单元
- [[dcrnn]] — DCRNN，扩散卷积 RNN（同年 ICLR 2018，RNN 系 STG 路线）
- [[diffstg]] — DiffSTG，扩散概率时空图预测（继承 GLU+时空块）
- [[specstg]] — SpecSTG，谱域扩散概率预测
- [[spatio-temporal-foundation-model]] — 时空基础模型概念
- [[urbandit]] — UrbanDiT，通用时空基础模型（STGCN 的远代后继）
- [[uniflow]] — UniFlow，统一 grid+graph 时空基础模型，STGCN 被用作 graph baseline (arXiv 2024)
- [[source-astgcn]] — ASTGCN，注意力增强图卷积（同年后续）
- [[mtgnn]] — MTGNN，自适应图学习范式（解决 STGCN 图不可学局限）
- [[source-conformer]] — ConFormer，事故感知交通预测
- [[hifinet]] — HiFiNet，层次频率分解 GNN 道路网络表示学习（AAAI 2026）
- [[road-network-representation-learning]] — 道路网络表示学习概念
- [[graph-frequency-decomposition]] — 图频率分解
- [[std-mae]] — STD-MAE，时空解耦预训练

[^src-stgcn]: [[source-stgcn]]
