---
title: "Gated Temporal Attention"
type: technique
tags:
  - self-attention
  - time-series
  - data-imputation
  - diffusion-models
  - gated-activation
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Gated Temporal Attention (GTA)

**GTA** (Gated Temporal Attention) 是 [[sadi|SADI]] 中用于建模跨特征时间依赖关系的残差注意力模块[^src-sadi]。受 DiffWave 和 WaveNet 残差块架构启发，GTA 将膨胀卷积替换为自注意力层，引入灵活性和自适应性以捕获非局部时序关系，特别适合缺失数据场景[^src-sadi]。

## 机制

GTA 由 $N_{GTA}$ 层堆叠组成，所有层的 skip connections 聚合为最终输出[^src-sadi]：

$$\tilde{X}, W_L, \epsilon'_n = \begin{cases} GTA_n(\hat{X}_{pos_i}, X_{pos}^{co}, t_{emb}) & n=1 \\ GTA_n(\tilde{X}, X_{pos}^{co}, t_{emb}) & \text{otherwise} \end{cases}$$

$$\epsilon_1 = \text{linear}\left(\frac{\sum_{n=1}^{N_{GTA}} \epsilon'_n}{\sqrt{2}}\right)$$

每层 GTA 包含：
1. **Self-Attention on Time Dimension**：在输入序列上加时间位置编码后进行时间维度的自注意力
2. **GLU (Gated Linear Unit) 激活**：对最后一层自注意力输出施加门控激活，控制信息流
3. **残差连接 + 扩散步嵌入注入**：$t_{emb}$ 作为条件信号注入

## 与 DiffWave/WaveNet 的关系

| 组件 | DiffWave/WaveNet | SADI GTA |
|------|-----------------|----------|
| 长程依赖 | 膨胀卷积 | **自注意力** |
| 局部模式 | 固定感受野 | **自适应全局感受野** |
| 灵活性 | 固定膨胀模式 | 数据驱动的注意力权重 |
| 缺失数据处理 | 需要完整序列 | **注意力天然跳过缺失位置** |

用自注意力替代膨胀卷积的动机：在缺失数据场景中，有效的上下文窗口可能是不规则的——自注意力可以根据数据内容动态调整关注范围，而非依赖固定的膨胀卷积感受野[^src-sadi]。

## 双 GTA 块设计

SADI 在去噪函数中使用两个 GTA 块[^src-sadi]：

| 块 | 输入 | 输出 | 目的 |
|----|------|------|------|
| GTA₁ | FDE 输出 + 观测值 + $t_{emb}$ | $\epsilon_1$ + 注意力权重 $W_L$ | 第一阶段插补 |
| GTA₂ | GTA₁ 的隐状态 + 重新引入的原始噪声数据 | $\epsilon_2$ | 第二阶段精炼 |

$W_L$ 从 GTA₁ 提取的注意力权重经 FFN + sigmoid 变换后作为两阶段输出的加权系数。

## 关联页面

- [[sadi]] — SADI，GTA 的宿主模型
- [[feature-dependency-encoder]] — FDE，GTA 的前置组件
- [[two-stage-imputation]] — GTA 双块设计所在的双阶段范式
- [[glu-gated-linear-unit]] — GLU 门控线性单元
- [[csdi]] — CSDI，其 Time Transformer 被 GTA 替代

[^src-sadi]: [[source-sadi]]
