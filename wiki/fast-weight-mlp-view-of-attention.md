---
title: "Fast-Weight MLP 视角下的 Attention"
type: concept
tags:
  - attention
  - fast-weights
  - efficient-attention
  - unified-framework
created: 2026-10-24
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Fast-Weight MLP 视角下的 Attention

MiTA 论文（arXiv:2602.01219）将 self-attention 形式化为「输入即时生成权重」的两层 MLP，据此构建高效注意力分类法（§3.1）[^src-2602-01219-mita]。

## 术语分工

论文区分两组权重（§2.1）[^src-2602-01219-mita]：

| 名称 | 载体 | 性质 |
|---|---|---|
| slow weights | 模型参数（含 FFN 的 $W_1,W_2$） | 编码持久知识，输入无关 |
| fast weights | 该层的 key-value 对 | 由输入条件生成，可视为短期记忆 |

在 attention 中，fast weights 指代 key-value 对；FFN 参数被视为 key-value 对对应的 slow weights（§1，引 Schlag et al. 2021、Geva et al. 2021）[^src-2602-01219-mita]。

## 形式化

普通 $D$ 宽慢权重两层 MLP（式 2）[^src-2602-01219-mita]：

$$\operatorname{MLP}_{\sigma}(x\mid W_1;W_2)=W_2\,\sigma\!\left(W_1^{\top}x+b_1\right)+b_2$$

scaled dot-product attention（式 1）可改写为 fast-weight 版本（式 3–4）[^src-2602-01219-mita]：

$$\operatorname{MLP}_{\exp}(q\mid\hat K;\hat V)=\hat V\exp\!\left(\hat K^{\top}q\right),\qquad \hat K=\frac{K}{\sqrt d},\quad \hat V=\frac{V}{\exp(\hat K^{\top}q)^{\top}\mathbf{1}_N}$$

该形式下，权重为缩放后的 keys 与 values，偏置为零，激活函数为 $\exp$。隐藏维度等于序列长度 $N$，慢权重 MLP 的宽度 $D$ 则与序列长度无关[^src-2602-01219-mita]。

## 「Unbounded fast-weight scaling」

由于隐藏宽度等于 $N$，单 query 计算开销随上下文线性增长，全序列累积为 $\mathcal{O}(N^2)$。论文将该现象表述为 unbounded fast-weight scaling，替代「二次复杂度」说法（§3.1）[^src-2602-01219-mita]。

此表述将降低复杂度的路径归结为控制隐藏宽度：压缩为更窄模块，或将 query 路由至部分宽度。论文据此将 Linear attention、Linformer、Agent attention、TTT、Deformable DETR、DAT、MoBA、NSA、Spark attention、DSA 纳入统一坐标，详见 [[fast-weight-scaling-taxonomy]]；MiTA 则兼具「压缩 + 路由」，详见 [[mita-attention]]。

## 这个视角与既有统一框架的分工

wiki 里已有两种统一注意力的写法，覆盖的家族不同，不能互相替代：

- [[linear-attention-unified-framework]] 为**公式级**统一，对齐 Mamba 选择性 SSM 与线性注意力的循环递推式，状态仅存在于线性注意力侧。
- [[cbsa]] 与 [[contract-and-broadcast-mechanism]] 为**算子级**统一，通过代表性 token 选取方式实例化 softmax / linear / channel / agent attention。
- fast-weight 视角为**结构级**统一：未给出精确等价公式，将各方法映射至降宽度的五维坐标。

三者在压缩分支交汇：CBSA 的代表性 token 与 MiTA 的 landmark query 均将宽度从 $N$ 压至 $m$；CBSA 采用 SVD 或固定基等矩阵分解，MiTA 采用 $Q$ 的分段平均池化[^src-2602-01219-mita]。

## 口径边界

- 式（3）的权重并非完全独立于 query：式（4）中 $\hat V$ 包含依赖 $q$ 的 softmax 归一化因子 $\exp(\hat K^{\top}q)^{\top}\mathbf{1}_N$。论文直接声明等价，未做限定[^src-2602-01219-mita]。
- 「宽度 $=N$」属于结构计数，与「物化 $N\times N$ 矩阵」不同：前者适用于所有 full attention，后者取决于算子实现。wiki 复杂度表（如 [[probsparse-self-attention]] 的 $O(L^2)$ 行）指代后者，两者不可混淆。

## 相关页面

- [[fast-weight-scaling-taxonomy]] — 由该视角导出的五维分类法与方法映射
- [[mita-attention]] — 压缩 + 路由的实例
- [[deformable-fast-weight-experts]] — 专家构造维的展开
- [[linear-attention-unified-framework]] — 公式级统一（Mamba ↔ 线性注意力）
- [[cbsa]] · [[representative-token-extraction]] — 压缩支的算子级统一
- [[performer]] — 核随机特征路线，在本视角下属「压缩成单个共享线性层」
- [[linear-attention-rnn-form]] — 递归状态 $s_i$ 即矩阵形式的 fast weights：核化线性注意力的原始机制页

[^src-2602-01219-mita]: [[source-2602-01219-mita]]
