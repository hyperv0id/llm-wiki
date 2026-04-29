---
title: "Temporal Rotation"
type: concept
tags:
  - positional-encoding
  - temporal-modeling
  - rotary-position-embedding
  - continuous-time
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Temporal Rotation

**时间旋转** 是指将连续时间戳（非离散序数位置）映射为 RoPE 旋转角的技术，使注意力机制能够直接建模时间维度的依赖关系[^src-siren-rope]。

## 背景

标准 RoPE 使用序数位置索引 $p_i$ 计算旋转角：
$$\theta_j = base^{-2j/d_k}$$
$$\text{RoPE}(x_m, m) = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} x_m$$

这种方法的问题：
- **序数是时间的粗劣代理**：7 天前的交互与 7 分钟前的交互在语义上完全不同
- **固定周期结构**：正弦振荡在自然语言中有意义，但对推荐系统无实际语义[^src-siren-rope]

## 时间旋转的洞见

类比复数平面：
- **实轴** → 语义维度（embedding）：编码"token 是什么"
- **虚轴** → 动态维度（旋转）：编码"token 如何与其他 token 关联"

旋转流形是一个未被充分探索的注意力表达维度[^src-siren-rope]

## SIREN-RoPE 实现

$$\Theta_j(T_i) = f_\phi(T_i)_j \cdot \omega_j^s$$

其中 $f_\phi$ 是双分支 SIREN-DNN 网络，将时间戳映射为每维旋转角[^src-siren-rope]

## 学到的周期性

实验发现模型自动学习到：
1. **日周期** (24h)：用户日内活跃模式
2. **周周期** (7天)：工作日/周末行为差异
3. **年衰减**：长期近因效应（DNN 分支）[^src-siren-rope]

## 与其他方法的对比

| 方法 | 时间信号类型 | 表达能力 |
|------|--------------|----------|
| 标准 RoPE | 序数位置 | 固定周期正弦 |
| TO-RoPE | 归一化时间戳 | 固定逆频率 |
| **时间旋转** | 连续时间戳 + SIREN | 可学习周期结构 |

## 相关页面

- [[siren-rope]] — 主 entity 页面
- [[dual-branch-siren]] — 架构详情
- [[ordinal-temporal-fusion]] — 序数-时间融合
- [[learnable-frequency-scaling]] — 可学习频率缩放

[^src-siren-rope]: [[source-siren-rope]]