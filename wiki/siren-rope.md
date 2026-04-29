---
title: "SIREN-RoPE"
type: entity
tags:
  - model
  - positional-encoding
  - rotary-position-embedding
  - temporal-modeling
  - sequential-recommendation
  - linkedin
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# SIREN-RoPE

SIREN-RoPE (Sinusoidal Representation Network Rotary Position Embedding) 是 LinkedIn 研究团队提出的一种新型旋转位置编码方法，将 RoPE 的固定旋转流形扩展为可学习的、时间信号条件化的空间[^src-siren-rope]。

## 核心思想

将 token embedding 视为**语义维度（实部）**，旋转流形视为**动态关系维度（虚部）**。语义编码"token 是什么"，旋转编码"token 如何与其他 token 在时间、位置和上下文上关联"[^src-siren-rope]。

## 与现有 RoPE 扩展的对比

| 方法 | 时间信号 | 可学习频率 | 序数门控 |
|------|----------|------------|----------|
| 标准 RoPE | ❌ | ❌ | — |
| TO-RoPE | ✅ (归一化时间戳) | ❌ | — |
| **SIREN-RoPE** | ✅ (SIREN-DNN) | ✅ | ✅ |

## 技术细节

### 旋转角度公式

$$\Theta_j(T_i, p_i) = f_\phi(T_i)_j \cdot \omega_j^s + p_i \cdot \theta_j \cdot \lambda$$

- $f_\phi$: 双分支 SIREN-DNN 网络
- $\omega_j^s$: 可学习每维频率缩放
- $\lambda$: 序数门控标量，初始化 1.0，训练后降至 0.044
- $\theta_j = base^{-2j/d_k}$: 标准 RoPE 逆频率[^src-siren-rope]

### 双分支架构

1. **周期性分支** ($f_{sin}$): SIREN 架构，sin 激活函数，自主发现隐藏的时间周期
2. **非周期性分支** ($f_{DNN}$): 标准 ReLU MLP，捕获单调趋势（近因衰减）[^src-siren-rope]

## 实验结果

在 LinkedIn 生产的社交信息流数据集上，SIREN-RoPE 在三个参与度任务（Contribution、Like、LongDwell）上均优于基线 Ordinal RoPE：

- Contribution NE: 0.6206 → 0.6182 (-0.0024)
- Like AUC: 0.9238 → 0.9249 (+0.0011)
- LongDwell AUC: 0.7597 → 0.7633 (+0.0036)

额外参数量仅约 0.2%，训练时间增加约 1.4%[^src-siren-rope]。

## 学习到的周期性

实验发现 SIREN-RoPE 从数据中自动学习到：
- **日周期** (24小时)：用户日内活跃模式
- **周周期** (7天)：工作日/周末行为差异
- **年衰减**：长期近因衰减（由 DNN 分支捕获）[^src-siren-rope]

## 代码

- 官方实现：https://github.com/hailingc/siren_rope

## 相关页面

- [[source-siren-rope]] — 论文 source-summary
- [[yarn]] — 另一种 RoPE 上下文扩展方法
- [[alibi]] — 另一种位置编码方法
- [[generalized-positional-encoding-framework]] — 统一位置编码理论框架

[^src-siren-rope]: [[source-siren-rope]]