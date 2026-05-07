---
title: "Efficient Cosine Operator (ECO)"
type: technique
tags:
  - graph-convolution
  - computational-efficiency
  - cosine-similarity
  - traffic-forecasting
  - linear-complexity
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# Efficient Cosine Operator (ECO)

ECO 是一种基于余弦相似度的线性复杂度图卷积算子，由 RAGC 提出，将自适应图卷积的计算复杂度从 O(N²) 降至 O(N)[^src-ragc-efficient-traffic-forecasting]。

## 动机

传统自适应图卷积构建自适应邻接矩阵 $A_{adp} = \text{softmax}(\text{ReLU}(E_{node}E_{node}^T))$，然后执行 $A_{adp} H$ 运算。由于 softmax 和 ReLU 不可分解，无法重排矩阵乘法顺序，导致必须显式构建 N×N 矩阵，复杂度为 O(N²)[^src-ragc-efficient-traffic-forecasting]。

BigST 使用随机特征映射近似 softmax 使其可分解，但引入近似噪声，影响训练稳定性和预测性能[^src-ragc-efficient-traffic-forecasting]。

## 核心思路

ECO 的关键洞察：**自适应邻接矩阵的本质是度量节点相似性，这可以通过余弦相似度有效捕捉——无需非线性激活**[^src-ragc-efficient-traffic-forecasting]。

## 三步计算

### 1. 门控机制

$$E_g = \text{softmax}(E_{node} W_1) \odot \text{ReLU}(E_{node} W_2)$$

- softmax 强调最相关邻居
- ReLU 移除弱连接（负值），确保邻接矩阵非负
- 门控通过 $W_1, W_2 \in \mathbb{R}^{d_{node} \times d_{node}}$ 以数据驱动方式学习连接模式

### 2. 余弦相似度矩阵

$$S = \hat{E}_g \hat{E}_g^T, \quad \hat{E}_g = \frac{E_g}{\|E_g\|_2}$$

ℓ2 归一化后内积 = 余弦相似度。自相似度始终为 1，有意保留以维持节点自身特征。

### 3. 矩阵结合律重排

行归一化后：

$$A_{adp} H^{(l)}_{mlp} = D^{-1} \hat{E}_g (\hat{E}_g^T H^{(l)}_{mlp})$$

$$D = \text{diag}(\hat{E}_g (\hat{E}_g^T \mathbf{1}_N))$$

关键：**不显式构建 N×N 矩阵**，直接用 $\hat{E}_g$ 参与计算。

## 复杂度对比

| 方法 | 图卷积复杂度 | 近似噪声 |
|------|-------------|---------|
| 传统 softmax 自适应 | O(N²·d₀) | 无 |
| BigST (随机特征映射) | O(N·φ·d₀) | 有 |
| ECO (余弦相似度) | O(N·d₀·d_node) | 无 |

当 d₀ 和 d_node 为固定超参数时，ECO 复杂度关于节点数 N 线性。

## 与其他线性化方法的区别

| 维度 | ECO | BigST | PatchSTG |
|------|-----|-------|----------|
| 复杂度 | O(N) | O(N) | O(N) |
| 近似噪声 | 无 | 有（随机映射） | 无 |
| 空间完整性 | 完整保留 | 完整保留 | 静态分区可能分割相邻节点 |
| 邻接矩阵构建 | 余弦相似度 | softmax 近似 | KD-tree 分块 |

## 相关页面

- [[ragc]] — ECO 所在的完整模型
- [[stochastic-shared-embedding|SSE]] — 配合使用的正则化技术
- [[residual-difference-mechanism|RDM]] — 噪声抑制机制
- [[large-scale-spatial-temporal-graph]] — 大规模时空图预测场景
- [[adaptive-graph-agent-attention|AGA-Att]] — 另一种空间复杂度优化方法

[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
