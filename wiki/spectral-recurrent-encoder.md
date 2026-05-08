---
title: "Spectral Recurrent Encoder (SG-GRU)"
type: technique
tags:
  - graph-neural-networks
  - spectral-methods
  - recurrent-networks
  - spatio-temporal
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Spectral Recurrent Encoder (SG-GRU)

SG-GRU（Spectral Graph GRU）是 [[specstg|SpecSTG]] 中提出的谱域图 GRU 编码器，在图傅里叶域编码历史时空信息，作为扩散过程的条件输入[^src-2401-08119-specstg]。它是 SpecSTG 将时空信息从原始域迁移到谱域的核心编码组件。

## 背景：标准 Graph GRU

### GRU 基础

标准 GRU（Gated Recurrent Unit）通过门控机制控制信息流[^src-2401-08119-specstg]：

$$\mathbf{r}_t = \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + b_r) \quad \text{(reset gate)}$$
$$\mathbf{z}_t = \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + b_z) \quad \text{(update gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(W_h \mathbf{x}_t + U_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + b_h) \quad \text{(new memory)}$$
$$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t \quad \text{(hidden state)}$$

### Graph GRU 的扩展

Graph GRU 在 GRU 的线性变换前加入图卷积，使得隐状态更新不仅依赖当前节点的输入，还聚合邻居节点的信息[^src-2401-08119-specstg]：

$$\mathbf{x}'_t = \text{GraphConv}(\mathbf{x}_t, A)$$
$$\mathbf{h}'_{t-1} = \text{GraphConv}(\mathbf{h}_{t-1}, A)$$

然后使用 $\mathbf{x}'_t$ 和 $\mathbf{h}'_{t-1}$ 替换原始 GRU 中的 $\mathbf{x}_t$ 和 $\mathbf{h}_{t-1}$。

**问题**：Graph GRU 中的图卷积复杂度为 $O(KN^2)$，在多步展开中累积为显著的计算瓶颈[^src-2401-08119-specstg]。

## SG-GRU 的设计

### 核心思想

SG-GRU 将 Graph GRU 的操作迁移至谱域[^src-2401-08119-specstg]：

1. **输入映射**：历史时间序列 $\mathbf{X}_{t-P:t}$ 通过图傅里叶变换映射到谱域 $\hat{\mathbf{X}}_{t-P:t} = U^\top \mathbf{X}_{t-P:t}$
2. **谱域门控**：在谱域执行 [[fast-spectral-graph-convolution|Fast Spectral Graph Convolution]] + GRU 门控更新
3. **输出条件**：输出谱域隐状态 $\mathbf{h}$，作为扩散去噪网络的条件输入

### 门控更新公式

在谱域中，SG-GRU 的门控更新使用 [[fast-spectral-graph-convolution|Fast Spectral GC]] 替代标准图卷积[^src-2401-08119-specstg]：

$$\hat{\mathbf{r}}_t = \sigma(\text{FastSpectralGC}(W_r \hat{\mathbf{x}}_t) + \text{FastSpectralGC}(U_r \hat{\mathbf{h}}_{t-1}) + b_r)$$
$$\hat{\mathbf{z}}_t = \sigma(\text{FastSpectralGC}(W_z \hat{\mathbf{x}}_t) + \text{FastSpectralGC}(U_z \hat{\mathbf{h}}_{t-1}) + b_z)$$
$$\tilde{\hat{\mathbf{h}}}_t = \tanh(\text{FastSpectralGC}(W_h \hat{\mathbf{x}}_t) + \text{FastSpectralGC}(U_h (\hat{\mathbf{r}}_t \odot \hat{\mathbf{h}}_{t-1})) + b_h)$$
$$\hat{\mathbf{h}}_t = \hat{\mathbf{z}}_t \odot \hat{\mathbf{h}}_{t-1} + (1 - \hat{\mathbf{z}}_t) \odot \tilde{\hat{\mathbf{h}}}_t$$

由于输入已在谱域，每个 [[fast-spectral-graph-convolution|Fast Spectral GC]] 调用的复杂度为 $O(KN)$ 而非 $O(KN^2)$[^src-2401-08119-specstg]。

### 复杂度对比

| 组件 | 标准 Graph GRU | SG-GRU |
|------|---------------|--------|
| 图卷积 | $O(KN^2)$ | $O(KN)$ |
| P 步展开 | $O(PKN^2)$ | $O(PKN)$ |
| 特征分解 | 不需要 | 预计算 $O(N^3)$（一次性） |

对于 $N = 307$（PEMS04）和 $P = 12$，SG-GRU 的编码复杂度比标准 Graph GRU 低约 $N = 307$ 倍[^src-2401-08119-specstg]。

## 在 SpecSTG 中的角色

### 条件编码器

SG-GRU 作为 [[specstg|SpecSTG]] 的条件编码器，将历史观测 $\mathbf{X}_{t-P:t}$ 编码为谱域隐状态 $\mathbf{h}$，用于引导扩散去噪网络的条件生成过程[^src-2401-08119-specstg]。

### 信息流

```
历史观测 X_{t-P:t}
       │ GFT
       ▼
谱域输入 X̂_{t-P:t}
       │ SG-GRU (P 步展开)
       ▼
条件隐状态 h ∈ R^{N×d}
       │
       ▼
注入 Spectral Graph WaveNet 去噪网络
```

### 与扩散过程的连接

SG-GRU 输出的隐状态 $\mathbf{h}$ 通过以下方式条件化去噪网络[^src-2401-08119-specstg]：

1. **时间嵌入**：扩散时间步 $s$ 通过位置嵌入编码
2. **条件注入**：$\mathbf{h}$ 与噪声谱表示 $\hat{\mathbf{y}}_s$ 和时间嵌入拼接，作为去噪网络的输入
3. **空间感知**：由于 $\mathbf{h}$ 在谱域，它自然编码了全局空间结构信息，使得条件生成过程能够利用图的空间拓扑

## 与其他时空编码器的对比

| 编码器 | 操作域 | 图卷积复杂度 | 空间信息 | 适用场景 |
|--------|--------|------------|---------|---------|
| Graph GRU | 原始域 | $O(KN^2)$ | 局部（邻居聚合） | 原始域扩散 |
| Graph Transformer | 原始域 | $O(N^2)$ | 全局（注意力） | 注意力模型 |
| [[spectral-recurrent-encoder\|SG-GRU]] | **谱域** | **$O(KN)$** | **全局（谱域自然嵌入）** | **谱域扩散** |

### 优势

1. **高效**：$O(KN)$ 图卷积使编码速度大幅提升[^src-2401-08119-specstg]
2. **全局空间感知**：谱域隐状态自然包含全局空间结构信息，无需额外的空间注意力机制
3. **与谱扩散无缝衔接**：编码器输出已在谱域，直接作为去噪网络的条件输入，无需域转换

### 局限性

1. 依赖静态图结构的特征分解，无法适应动态图
2. GRU 的序列展开在长历史序列上仍有 $O(P)$ 的深度，不如 Transformer 可并行化
3. 谱域隐状态的语义解释性不如原始域直观

## 关联页面

- [[specstg]] — 使用此技术的概率时空图预测框架
- [[fast-spectral-graph-convolution]] — SG-GRU 中使用的快速谱图卷积
- [[spectral-graph-wavelet-transform]] — 谱图小波变换
- [[traffic-forecasting]] — 时空图交通预测总览

[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
