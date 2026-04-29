---
title: "Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling"
type: source-summary
tags:
  - positional-encoding
  - rotary-position-embedding
  - temporal-modeling
  - sequential-recommendation
  - siren
  - continuous-time
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling

## 核心论点

本文提出**SIREN-RoPE**，将 RoPE 的旋转流形从固定的序数位置索引扩展为可学习的、时间信号条件化的空间[^src-siren-rope]。核心洞见是：token embedding 编码语义（实部），旋转编码动态关系（虚部）——旋转流形是一个未被充分探索的注意力表达维度。

## 主要贡献

1. **Temporal Rotation**：使用双分支 SIREN-DNN 网络将连续时间戳映射为每维旋转角，同时捕获周期性（日/周）和非周期性（近因衰减）时间结构[^src-siren-rope]
2. **Adaptive Frequency Learning**：每维频率尺度和序数门控 λ 可联合学习，替代手调的逆频率常数[^src-siren-rope]
3. **Empirical Validation**：在生产级社交信息流数据集上，SIREN-RoPE 在三个参与度任务（Contribution、Like、LongDwell）的校准（NE）和排序（AUC）指标上均取得一致提升，额外参数量仅约 0.2%[^src-siren-rope]

## 与现有工作的关系

- **标准 RoPE**：使用固定逆频率 schedule $\theta_j = base^{-2j/d_k}$ 从序数位置计算旋转角[^src-siren-rope]
- **TO-RoPE** (Wei et al., 2025)：将序数替换为归一化时间戳，但仍使用固定逆频率[^src-siren-rope]
- **SIREN-RoPE** 泛化了 TO-RoPE，用可学习的双分支 SIREN-DNN 替代固定频率映射

## 关键公式

### 序数-时间融合

$$\Theta_j(T_i, p_i) = f_\phi(T_i)_j \cdot \omega_j^s + p_i \cdot \theta_j \cdot \lambda$$

其中 $f_\phi$ 是双分支 SIREN 网络，$\omega_j^s$ 是可学习频率缩放，$\lambda$ 是序数门控（初始化 1.0，训练后降至 0.044）[^src-siren-rope]

### 双分支架构

$$f_\phi(T) = f_{sin}(T) + f_{DNN}(T)$$

- 周期性分支 $f_{sin}$：SIREN 结构，sin 激活函数，自主发现时间周期性
- 非周期性分支 $f_{DNN}$：标准 ReLU MLP，捕获单调趋势（如近因衰减）[^src-siren-rope]

### 时间输入特征

$$t(T) = [\cos(2\pi T/\tau_d), \sin(2\pi T/\tau_d), \cos(2\pi T/\tau_w), \sin(2\pi T/\tau_w), \tilde{T}]$$

其中 $\tau_d=86400s$（日周期），$\tau_w=604800s$（周周期），$\tilde{T}$ 是归一化长期偏移[^src-siren-rope]

## 实验结果

| Model | Contribution NE | Contribution AUC | Like NE | Like AUC | LongDwell NE | LongDwell AUC |
|-------|-----------------|------------------|---------|----------|--------------|---------------|
| Ordinal RoPE | 0.6206 | 0.9102 | 0.5985 | 0.9238 | 0.8362 | 0.7597 |
| Timestamp-as-Feature | 0.6218 | 0.9098 | 0.5997 | 0.9233 | 0.8350 | 0.7603 |
| TO-RoPE | 0.6218 | 0.9095 | 0.5999 | 0.9231 | 0.8349 | 0.7613 |
| **SIREN-RoPE** | **0.6182** | **0.9115** | **0.5963** | **0.9249** | **0.8334** | **0.7633** |

SIREN-RoPE 在所有 6 个指标上均优于基线 Ordinal RoPE，最显著提升在 LongDwell AUC (+0.0036)[^src-siren-rope]。

## 消融实验发现

1. **周期性特征提升校准**：使用 (cos, sin) 分解优于标量 time_in_year[^src-siren-rope]
2. **SIREN 分支的价值**：在已知日/周周期的情况下，DNN-only 达到最佳 NE，但保留 SIREN 分支可自主发现隐藏周期[^src-siren-rope]
3. **语义旋转无效**：静态信号（如 in/out-of-network）放在旋转维度不如放在 embedding 维度[^src-siren-rope]

## 局限性

- 外推约束：模型在训练时未见的时间间隔或全新语义类别上的泛化能力待验证[^src-siren-rope]
- 训练后的 λ=0.044 表明序数信号几乎被时间信号完全替代[^src-siren-rope]

## 参考文献

[^src-siren-rope]: [[source-siren-rope]]