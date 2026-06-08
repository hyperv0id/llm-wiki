---
title: "Spatio-Temporal Retrieval Store"
type: technique
tags:
  - retrieval-augmented
  - faiss
  - memory-bank
  - spatio-temporal
  - indexing
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Spatio-Temporal Retrieval Store

时空检索存储（ST-Retrieval Store）是 [[rast|RAST]] 框架的核心组件，是一个基于 FAISS 的双维度向量记忆库，用于维护和检索细粒度时空历史模式。[^src-rast]

## 结构

检索存储维护两个独立的记忆库：[^src-rast]

$$M = \{M_{sp}, M_{tp}\}$$

- **空间记忆库** $M_{sp}$：存储空间嵌入向量 $\{v_{sp}^{(i)}, m_{sp}^{(i)}\}$，其中 $m^{(i)}$ 包含统计摘要和重要性度量
- **时间记忆库** $M_{tp}$：存储时间嵌入向量 $\{v_{tp}^{(j)}, m_{tp}^{(j)}\}$

## 索引与检索

### FAISS 索引

使用 FAISS（Facebook AI Similarity Search）库执行高效的相似度搜索。[^src-rast] 索引支持三种优化：
1. **定期重建**（每 10 个 epoch）
2. **LRU 缓存**
3. **GPU 加速**

检索复杂度 $O(k \log M + kd)$（IVF 倒排文件索引），远低于图注意力 $O(N^2)$。[^src-rast]

### L2 距离检索

给定查询 $Q$ 和索引 $I$，ST-Retriever 通过 L2 距离执行 Top-k 检索：

$$D(Q, v_i) = -\|Q - v_i\|_2^2$$

$$\text{Retriever}(Q, I, k) = \arg\max_k\{D(Q, v_j)\}_{j=1}^{|V|}$$

### 信息论评分

检索到的向量权重由两部分组成：[^src-rast]
- **相似度分数** $s_i = D(Q, v_i)$
- **动量分数** $\omega_i$：基于信息熵 $H(v) = -\sum_{d=1}^D p_d \log p_d$ 和多样性-相似度系数 $\lambda$ 更新：

$$\omega_i' = \omega_i + \text{softmax}((s_i + \lambda \cdot H(v_i))/\tau)$$

## 记忆管理

### 动量更新

记忆库在每个 epoch 以指数移动平均（EMA）更新：[^src-rast]

$$M_s^{(e+1)} = (1 - \omega_s)M_s^{(e)} + \omega_s \cdot \sigma(E_s)$$

其中 $\sigma(\cdot)$ 为插入操作，$\omega_s$ 由相似度分数自适应确定。

### 容量管理

实施混合策略以防止无界增长：[^src-rast]
- **时间衰减**：50 个 epoch 以上的旧模式被衰减
- **相似度剪枝**：低于 0.3 相似度阈值的低质量模式被淘汰
- **容量限制**：每个记忆库最多 1000 个模式

## 与相关检索技术的对比

| 方法 | 检索维度 | 存储 | 距离度量 | 更新策略 |
|------|----------|------|----------|----------|
| RAST | 时间+空间双维度 | FAISS IVF | L2 | 动量 EMA |
| [[gtr|GTR]] | 仅时间 | 可学习参数 Q | 索引定位 | 梯度优化 |
| [[uniflow|UniFlow]] ST-MRA | 时间/频域四库 | 结构化记忆 | 余弦相似度 | 可学习 |

[^src-rast]: [[source-rast]]