---
title: "ALiBi"
type: entity
tags:
  - transformer
  - position-embedding
  - extrapolation
  - iclr-2022
created: 2026-04-28
last_updated: 2026-10-26
source_count: 5
confidence: high
status: active
---

# ALiBi (Attention with Linear Biases)

ALiBi 是由 Ofir Press、Noah A. Smith、Mike Lewis 提出的一种位置编码方法（ICLR 2022），首次实现了 Transformer 模型在训练短序列后能高效外推到更长序列进行推理[^src-alibi]。

## 核心思想

ALiBi 不在词嵌入中添加位置编码，而是在计算注意力分数后，添加一个与 query-key 距离成线性关系的偏置[^src-alibi]：

$$\text{softmax}(q_i K^T + m \cdot [-(i-1), ..., -2, -1, 0])$$

其中 $m$ 是每个注意力头的斜率（slope），为固定值而非学习参数[^src-alibi]。

## 几何斜率序列

对于有 $n$ 个头的模型，斜率采用几何序列[^src-alibi]：

$$2^{-8/n}, 2^{-7/n}, ..., 2^0$$

- 8 头模型：$2^{-1}, 2^{-2}, ..., 2^{-8}$
- 16 头模型：通过对相邻斜率几何平均插值

这种预设的斜率序列可泛化到不同模型规模和数据集，无需针对新任务调参[^src-alibi]。

## 特性

| 特性 | 描述 |
|------|------|
| **归纳偏置** | 对近邻位置有天然偏好（recency）[^src-alibi] |
| **无位置嵌入** | 不添加任何位置编码到词嵌入[^src-alibi] |
| **相对位置** | 每层注入位置信息到 Q 和 K，不注入 V[^src-alibi] |
| **固定斜率** | 斜率不参与训练，保持固定[^src-alibi] |

## 时序预测中的扩展

ALiBi 的斜率与"加性偏置"形式被搬到时间序列预测，改动集中在距离项：

| 做法 | 距离项 | 出处 |
|---|---|---|
| 原式 | $m_k\cdot\lvert i-j\rvert$ | [[linear-attention-bias\|线性注意力偏置]] |
| 周期折叠 | $m_k\cdot\min\!\left(\lvert i-j\rvert \bmod P_S,\ P_S-\lvert i-j\rvert\right)$ | [[periodic-nested-attention-bias\|周期化相对注意力偏置]]（PENGUIN，AISTATS 2026）[^src-penguin] |
| 局部性先验 | 距离衰减用于金融时序的局部归纳偏置 | TIPS（AAAI 2026）[^src-tips] |

PENGUIN 把斜率公式 $m_k=2^{-k/8}$ 直接沿用、固定不训练，只在多头之上再加一层"组"：每组绑定一个周期长度并共享该组的 $K,V$，从而在一次注意力里同时处理多个周期[^src-penguin]。它的消融显示偏置项本身贡献最大（Traffic avg MSE 0.407→0.396），周期形式再降至 0.390[^src-penguin]。细节见 [[penguin|PENGUIN]]。

## 性能对比

### 外推能力

| 方法 | L=512 训练最大外推长度 |
|------|----------------------|
| Sinusoidal | ~560 tokens |
| Rotary | ~700 tokens |
| T5 Bias | ~1100 tokens |
| **ALiBi** | **>12000 tokens** |

### 效率

| 指标 | Sinusoidal | ALiBi |
|------|-----------|-------|
| 训练速度 | 基准 | +11% |
| 内存使用 | 基准 | +0~100MB |
| 额外参数 | 0 | 0 |

## 应用场景

- **长上下文推理**：训练短序列，外推处理更长序列[^src-alibi]
- **Few-shot learning**：在 NLP 任务中可输入更多示例[^src-alibi]
- **长文本生成**：支持生成长于训练序列的输出[^src-alibi]

## 与其他位置编码的关系

- **Sinusoidal**：绝对位置编码，无法外推[^src-alibi]
- **Rotary**：相对位置编码，部分外推能力但有额外计算开销[^src-alibi]
- **T5 Bias**：相对位置编码，外推效果好但训练速度慢 2 倍[^src-alibi]
- **ALiBi**：相对位置编码，高效外推，无额外运行时开销[^src-alibi]
- **[[relative-position-representation|相对位置表示]]（Shaw et al. 2018）**：$2k+1$ 个可学习向量按 $\operatorname{clip}(j-i,k)$ 加到 $K$ 与 $V$ 上，向量随头学习；ALiBi 把这条路径压成每个头一个固定斜率、只改 logit、不进 value[^src-shaw-2018-relative-position][^src-alibi]

## 局限性

- 外推性能在约 2L 达到峰值，之后逐渐下降[^src-alibi]
- 论文本身只验证了语言任务，时序预测上的迁移由后续工作完成（见上节）[^src-alibi]
- 在 [[generalized-positional-encoding-framework|GPE 框架]]视角下，ALiBi 无法满足 [[long-distance-correlation-preservation|LDCP]] 和 [[gradient-positional-sensitivity|GPS]]，导致远距离相关性被完全抑制且梯度不携带位置信息[^src-vetcha-2026-towards-infinite-length-extrapolation]

## 引用

[^src-alibi]: [[source-alibi]]
[^src-penguin]: [[source-penguin]]
[^src-tips]: [[source-tips]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]
[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]
