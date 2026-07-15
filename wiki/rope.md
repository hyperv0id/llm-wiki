---
title: "Rotary Position Embedding (RoPE)"
type: technique
tags:
  - position-encoding
  - transformer
  - attention
  - rotary-position-embedding
  - llm
created: 2026-06-22
last_updated: 2026-07-14
source_count: 2
confidence: high
status: active
---

# Rotary Position Embedding (RoPE)

RoPE（Rotary Position Embedding）是一种乘法型相对位置编码方法，由苏剑林等人在 2023 年提出[^src-roformer]。它已成为现代大语言模型的标准位置编码方案，被 LLaMA、Mistral、Qwen、DeepSeek 等模型广泛采用。

## 动机：从加法到乘法

传统位置编码（绝对位置嵌入、XLNet 式相对位置偏置）均基于 **加法** — 将位置信号加到词嵌入上。RoPE 的出发点是：**能否让自注意力的内积天然地只依赖相对位置？**

即寻找函数 $f_q, f_k$ 使得：

$$\langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n) \rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)$$

这在数学上等价于要求 query/key 在纳入位置信息后，其内积仅通过 $(m-n)$ 依赖于位置。[^src-roformer]

## 形式化

### 二维情形

在 $d=2$ 时，可以证明解为复数乘法（即二维旋转）：

$$f_q(\boldsymbol{x}_m, m) = (W_q \boldsymbol{x}_m) e^{im\theta}$$
$$f_k(\boldsymbol{x}_n, n) = (W_k \boldsymbol{x}_n) e^{in\theta}$$

其中 $\theta$ 为非零常数。这意味着：**将 query/key 向量按其绝对位置旋转相应角度**[^src-roformer]。

### 高维推广

将 $d$ 维空间分成 $d/2$ 个二维子空间，每对维度独立旋转：

$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}^d_{\Theta,m} W_{\{q,k\}} \boldsymbol{x}_m$$

其中旋转矩阵 $\boldsymbol{R}^d_{\Theta,m}$ 为分块对角矩阵，每块为 $2\times 2$ 旋转矩阵：

$$\boldsymbol{R}^d_{\Theta,m} = \text{diag}\left(
\begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 \\ \sin m\theta_1 & \cos m\theta_1 \end{pmatrix}, \cdots,
\begin{pmatrix} \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\ \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix}
\right)$$

频率参数 $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, d/2]\}$ 直接取自原始 Transformer 的正弦位置编码[^src-roformer]。

### 自注意力中的应用

代入自注意力公式得：

$$\boldsymbol{q}_m^\top \boldsymbol{k}_n = (\boldsymbol{R}^d_{\Theta,m} W_q \boldsymbol{x}_m)^\top (\boldsymbol{R}^d_{\Theta,n} W_k \boldsymbol{x}_n) = \boldsymbol{x}_m^\top W_q^\top \boldsymbol{R}^d_{\Theta, n-m} W_k \boldsymbol{x}_n$$

其中 $\boldsymbol{R}^d_{\Theta, n-m} = (\boldsymbol{R}^d_{\Theta,m})^\top \boldsymbol{R}^d_{\Theta,n}$。关键结果：**内积仅依赖相对位置 $(n-m)$**。[^src-roformer]

### 高效实现

由于 $\boldsymbol{R}^d_{\Theta,m}$ 的分块稀疏性，实际实现使用逐元素乘加而非矩阵乘法：

$$\boldsymbol{R}^d_{\Theta,m} \boldsymbol{x} = \boldsymbol{x} \otimes (\cos m\theta_1, \cos m\theta_1, \cdots, \cos m\theta_{d/2}, \cos m\theta_{d/2}) + \text{rot}(\boldsymbol{x}) \otimes (\sin m\theta_1, \sin m\theta_1, \cdots, \sin m\theta_{d/2}, \sin m\theta_{d/2})$$

其中 $\text{rot}(\boldsymbol{x}) = (-x_2, x_1, -x_4, x_3, \cdots, -x_d, x_{d-1})$。时间复杂度 $O(d)$。[^src-roformer]

## 核心性质

### 1. 长期衰减

当相对距离增加时，query-key 内积绝对值的上界减小，即远距离 token 间的注意力权重自然衰减。这由 $\theta_i$ 的指数衰减和 Abel 变换保证[^src-roformer]。

### 2. 序列长度外推

由于旋转矩阵 $\boldsymbol{R}^d_{\Theta,m}$ 对任意 $m$ 都有定义，RoPE 理论上可处理任意长度序列，不受训练时最大长度的限制。这为后续的上下文窗口扩展（如 [[yarn|YaRN]]、[[ntk-aware-interpolation|NTK-aware]] 等）奠定了基础。

### 3. 线性注意力兼容

旋转是保范变换（$\|\boldsymbol{R}\boldsymbol{x}\| = \|\boldsymbol{x}\|$），因此 RoPE 可直接与线性注意力结合：

$$\text{Attention}(Q, K, V)_m = \frac{\sum_n (\boldsymbol{R}^d_{\Theta,m} \phi(q_m))^\top (\boldsymbol{R}^d_{\Theta,n} \varphi(k_n)) \boldsymbol{v}_n}{\sum_n \phi(q_m)^\top \varphi(k_n)}$$

这是首个能与线性自注意力兼容的相对位置编码方案[^src-roformer]。

### 4. 相对位置但绝对值编码

RoPE 的独特之处：**形式上编码绝对位置（旋转角度），效果上产生相对位置依赖（内积仅依赖差值）**。区别于纯粹的绝对位置编码（无相对信息）和纯粹的相对位置偏置（无绝对信息）。

## 与其他位置编码的对比

| 方法 | 类型 | 外推能力 | 线性注意兼容 | 衰减机制 |
|------|------|----------|-------------|----------|
| 正弦位置编码 | 绝对加法 | ❌ | ❌ | ❌ |
| 可学习绝对位置 | 绝对加法 | ❌ (受 max_len 限制) | ❌ | ❌ |
| XLNet 式相对偏置 | 相对加法 | ✅ | ❌ | ❌ |
| [[alibi|ALiBi]] | 相对偏置 | ✅ | ✅ | ✅ (线性偏置) |
| **RoPE** | **绝对旋转→相对内积** | ✅ | ✅ | ✅ (乘法衰减) |

## 在 LLM 中的统治地位

RoPE 被几乎所有主流开源 LLM 采用：
- **LLaMA / LLaMA 2/3/4** (Meta)
- **Mistral / Mixtral** (Mistral AI)
- **Qwen / Qwen 2/2.5** (Alibaba)
- **DeepSeek / DeepSeek-V2/V3** (DeepSeek)
- **Gemma** (Google)
- **Phi-3/4** (Microsoft)

## 后续发展

RoPE 引发了大量后续研究，包括但不限于：

- **上下文扩展**：[[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]]、[[dynamic-scaling]]、[[yarn]]
- **时间序列适配**：[[temporal-rotation]]（SIREN-RoPE，将固定序数索引替换为可学习时间条件化旋转）、[[learnable-frequency-scaling]]（可学习频率缩放）
- **物理知情位置编码**：[[physics-informed-position-encoding|PIPE]] 将 RoPE 的位置 ID 替换为物理量（经纬度+时间），配合[[variant-frequency-positional-encoding|变频率正弦编码]]，在台风预测中实现 SOTA（NeurIPS 2025）[^src-pipe]
- **统一理论框架**：[[generalized-positional-encoding-framework]]
- **注意力温度**：[[attention-temperature-scaling]]（等价于缩放 RoPE 旋转嵌入）
- **收敛性分析**：[[convergent-normalization]]

## 相关页面

- [[source-roformer]] — 原始论文摘要
- [[roformer]] — RoFormer 模型实体
- [[siren-rope]] — 时间条件化 RoPE 扩展
- [[alibi]] — 替代位置编码方案
- [[scaling-factor-sqrt-dk]] — 注意力 $\frac{1}{\sqrt{d_k}}$ 缩放因子的数值稳定性
- [[physics-informed-position-encoding]] — PIPE：RoPE 的物理知情扩展
- [[pipe]] — PIPE 台风预测模型

[^src-roformer]: [[source-roformer]]
[^src-pipe]: [[source-pipe]]
