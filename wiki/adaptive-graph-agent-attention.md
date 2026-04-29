---
title: "Adaptive Graph Agent Attention (AGA-Att)"
type: technique
tags:
  - spatial-temporal
  - attention-mechanism
  - computational-efficiency
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
references:
  - [[source-fast-long-horizon-forecasting]]
confidence: medium
status: active
---

# Adaptive Graph Agent Attention (AGA-Att)

## 背景

传统时空图预测方法面临严重的计算瓶颈：
- **GCN/GNN**: O(N²) 节点交互复杂度
- **Self-Attention**: O(N²) 成对交互

当节点数 N 很大时（如 8600 节点），计算和内存成本爆炸性增长。

## 核心思想

引入一组**可学习的代理 tokens** (agent tokens) U = {u_i | i ∈ [1, a]}，其中 a ≪ N，作为节点之间信息传递的中介。

关键洞察：时空图中存在**空间冗余**——节点级别的时序行为包含共享模式，因此可以用少量潜在原型（latent prototypes）来近似节点表示流形。

## 两阶段注意力机制

### 1. Node-to-Agent Aggregation Attention

每个 agent token 查询所有图节点 token，聚合长程空间信息：

```
Q_A^��� = A W_{agg,1}^ℓ        # Agent 作为 Query
K_G^ℓ = H_t^ℓ W_{agg,2}^ℓ    # 节点作为 Key
A_agg = Softmax( (Q_A^ℓ) (K_G^ℓ)^T / √d )  ∈ R^(a×N)
```

### 2. Agent-to-Distribution Attention

每个图节点 token 查询 agent tokens 获取全局上下文，再分发回节点：

```
Q_G^ℓ = H_t^ℓ W_{dist,1}^ℓ   # 节点作为 Query
K_A^ℓ = A W_{dist,2}^ℓ       # Agent 作为 Key
A_dist = Softmax( (Q_G^ℓ) (K_A^ℓ)^T / √d )  ∈ R^(N×a)
```

### 3. 最终输出

```
V_t^{ℓ-1} = H_t^{ℓ-1} W_V^ℓ   # 节点作为 Value
AGA-Att(H_t^{ℓ-1}) = A_dist (A_agg V_t^{ℓ-1})
```

## 复杂度分析

| 模块 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 传统自注意力 | O(N²) | O(N²) |
| AGA-Att | O(N·a·d) | O(N·a) |

当 a = 32, d = 64 时：
- N = 8600: O(8600·32·64) ≈ 17.6M vs O(8600²) ≈ 74M
- 约 **4x** 计算节省

## 设计优势

1. **线性于节点数**：计算复杂度与 N 线性相关而非二次
2. **捕获长程依赖**：通过 agent tokens 间接实现任意节点间的信息交互
3. **自适应学习**：agent tokens 是可学习的，不依赖预设图结构
4. **空间冗余利用**：假设节点表示可用少量原型近似

## 与其他方法的对比

- **SGP**: 稀疏聚合，但依赖准确图结构
- **BigST**: 线性注意力 O(N·φ)，但仍需逐节点计算
- **STID**: 完全消除空间交互，丢失所有空间依赖
- **AGA-Att**: 用 a ≪ N 个代理 tokens 实现高效全局交互

## 在 FaST 中的应用

作为 backbone 的第一层模块，先于 HA-MoE 处理空间依赖：

```
H_t^ℓ = RMSNorm( HA-MoE(Z_t^ℓ, X_t) + Z_t^ℓ )
Z_t^ℓ = RMSNorm( AGA-Att(H_t^{ℓ-1}) + H_t^{ℓ-1} )
```

## 参数选择

- **Agent 数量 a**: 需要在效率和信息容量间权衡
  - a 太小：丢失空间信息
  - a 太大：退化为传统注意力
- FaST 默认 a = 32，在 8600 节点上效果良好

## 相关页面

- [[mixture-of-experts|MoE]] — FaST 中与 AGA-Att 配合使用的特征变换模块
- [[large-scale-spatial-temporal-graph]] — 大规模时空图预测场景

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]