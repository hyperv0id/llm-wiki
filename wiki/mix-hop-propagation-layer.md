---
title: "Mix-Hop Propagation Layer"
type: technique
tags:
  - graph-neural-network
  - spatial-dependency
  - information-propagation
  - over-smoothing
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Mix-Hop Propagation Layer

Mix-Hop Propagation Layer（Mix-Hop 传播层）是 [[mtgnn|MTGNN]] (KDD 2020) 提出的图卷积模块核心组件，设计目的是在有向图上传播节点信息的同时避免过平滑问题 [^src-mtgnn]。

## 机制

Mix-Hop 传播层由两步操作组成：

### 信息传播 (Information Propagation)

H^(k) = β·H_in + (1-β)·Ã·H^(k-1)

其中 β 为超参数，控制根节点原始状态的保留比例；Ã = D̃⁻¹(A+I) 为归一化邻接矩阵。该步骤沿图结构递归传播信息，但通过保留部分根节点状态避免了传统 GCN 中深层节点隐藏状态收敛到同一点的过平滑问题 [^src-mtgnn]。β 确保传播后的节点状态既保留局部性又能探索深层邻域 [^src-mtgnn]。

### 信息选择 (Information Selection)

H_out = Σ(k=0..K) H^(k)·W^(k)

对第 0 到 K 跳的传播结果进行加权求和，W^(k) 为可学习参数矩阵，功能相当于特征选择器 [^src-mtgnn]。当图结构不含空间依赖时，W^(k) 可被学习调至 0（k>0），保留节点自身信息不受噪声干扰 [^src-mtgnn]。

## 关键设计理由

- **避免过平滑**：β 保留根节点状态，阻止多跳传播后所有节点收敛到相同表示 [^src-mtgnn]
- **噪声过滤**：信息选择步骤通过可学习参数自适应过滤无意义的邻居信息 [^src-mtgnn]
- **高效表示相邻跳差分**：单层 mix-hop 即可表示相邻跳之间的差值（设 K=2, W(0)=0, W(1)=-1, W(2)=1 → H_out = H₂ - H₁），比拼接方法更高效 [^src-mtgnn]

## 在图卷积模块中的使用

MTGNN 的图卷积模块包含两个 mix-hop 传播层——一个处理入流信息（沿 A 传播），一个处理出流信息（沿 Aᵀ 传播），两者求和得到净流入信息 [^src-mtgnn]。

## 与前作的连接

Mix-hop 思想源自 MixHop (Kapoor et al., ICML 2019) 和 DAGCN (Chen et al., IJCNN 2019)，但 MTGNN 的改进在于：(1) 用信息传播 + 信息选择替代拼接/注意力聚合；(2) 引入 β 保持局部-全局平衡；(3) 单层即可表示跳间线性交互 [^src-mtgnn]。

## 相关页面

- [[mtgnn]] — MTGNN 模型（使用 Mix-Hop 的上下文）
- [[graph-learning-layer]] — 提供邻接矩阵的上游模块
- [[dilated-inception-layer]] — 并行的时间卷积模块

[^src-mtgnn]: [[source-mtgnn]]
