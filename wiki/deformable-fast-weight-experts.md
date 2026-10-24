---
title: "形变 Fast-Weight 专家（Deformable Experts）"
type: technique
tags:
  - attention
  - sparse-attention
  - mixture-of-experts
  - fast-weights
  - routing
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# 形变 Fast-Weight 专家

在 [[fast-weight-mlp-view-of-attention|fast-weight 视角]]下，将 $N$ 宽 MLP 拆分为专家存在两类机制：按位置切分固定块，或按内容聚合语义相关的 key-value 对。后者即形变专家：专家构成随输入动态变化，脱离固定位置。MiTA 论文（arXiv:2602.01219，§2.2、§3.2–3.3）提出并定义该形态[^src-2602-01219-mita]。

## 与刚性块的差别

[[moba-attention|MoBA]] 类方法将专家划分为连续、不重叠的定长块，通过块内 key 平均池化或参数化模块生成路由向量，专家划分与输入内容无关。论文把这种划分称为 coarse and suboptimal，并把「专家本身也应 query-aware」作为改造动机（§1、§2.2）[^src-2602-01219-mita]，类比以 deformable patch tokenization 替代固定矩形切块（DPT，Chen et al. 2021）[^src-2602-01219-mita]。

## 专家构造的三个条件源

| 条件源 | 代表 | 构造方式 |
|---|---|---|
| 仅 query | DAT | query 决定的共享专家 |
| 仅 key-value | MHLA | KV 侧先验（局部核化） |
| query 与 KV | MiTA | landmark query 对全 KV 打分后取 top-$k$ |

论文将 MoBA 归类为「位置驱动」（position-driven），排除在内容相关方法之外（§3.3）[^src-2602-01219-mita]。

## 两种「逐 query top-$k$」与 MiTA 的区别

Spark attention 通过部分特征轻量 skim、DSA 采用 FP8 低精度打分，二者均由**各个 query 独立**选取 top-$k$ 个 KV，导致专家数达到 $N$，单专家宽度为 $k$。论文指出该设计将隐藏宽度 $N$ 转化为专家数量 $N$（§1）[^src-2602-01219-mita]。

MiTA 改用**少量 landmark query** 各自选取 top-$k$，构建 $m$ 个可复用专家组，query 侧执行单次 top-1 路由。单 query 计算形状得以固定，代价为引入一次 $N$-to-$m$ 的 argmax 检索[^src-2602-01219-mita]。

## MiTA 里的具体形态

- **专家**：$\mathcal{E}_i(q)=\operatorname{SDPA}(q,K^{(i)},V^{(i)})$，其中 $K^{(i)}=K_{:,\mathcal{I}_i}$、$\mathcal{I}_i=\operatorname{Top}_k(K^{\top}\tilde q_i)$，$i=1,\dots,m$，专家宽度固定为 $k$（式 5–7）[^src-2602-01219-mita]。
- **共享专家**：landmark values $\tilde V=V\operatorname{softmax}(K^{\top}\tilde Q/\sqrt d)$，共享专家为 $\operatorname{SDPA}(q,\tilde Q,\tilde V)$，对全体 query 常开，起全局概括作用（式 8–9）。这一角色与 MoBA 强制选中当前块、类比 shared expert 的做法同构[^src-2602-01219-mita]。
- **路由**：landmark query 充当专家表征，$\operatorname{argmax}(Q^{\top}\tilde Q)$ 给出 query 的 top-1 目标专家；query 经专家排序后，通过变长 FlashAttention 一次处理 $m$ 个长度为 $k$ 的分段（Algorithm 1）[^src-2602-01219-mita]。
- **合并**：目标专家 KV 与共享专家 KV 拼接为 $[\,\tilde Q,\,K^{(e_1)}\,]$，通过 online softmax 合并；$s=1$ 时单 query 实际计算 $m+k$ 对 KV（默认 $m=k=25$，即 50 对）（式 10–12）[^src-2602-01219-mita]。

在 $s>1$ 时专家组合可构成 $\binom{m}{s}$ 种稀疏模式；论文指出 $s=1$ 存在退化为硬聚簇、阻碍簇间信息流通的潜在风险（§3.3）[^src-2602-01219-mita]。

## 证据状态

- Fig. 3：专家的 KV 与所路由 query 均超出 landmark query 原始采样窗口，表明专家非局部窗口复制[^src-2602-01219-mita]。
- Fig. 5：各层中「专家所选 KV 位置」与「路由 query 位置」的 mIoU 保持低位，作者据此判定 $s=1$ 为路由机制而非硬聚簇；图表无具体数值[^src-2602-01219-mita]。
- Fig. 4：深层多数 token 未被选为专家，注意力聚集于类别相关区域，作者称之为 token-pruning 效应；无量化剪枝比例与消融对比[^src-2602-01219-mita]。
- 缺失：缺少与 MoBA、NSA、VMoBA、MoGA 的直接对比，未提供 $m$、$k$、$s$ 在 MiTA 上的消融。形变专家的实际增益仅能由与纯压缩方法 Agent attention 的对比间接推断（详见 [[mita-attention]] 证据节）[^src-2602-01219-mita]。

## 相关页面

- [[mita-attention]] — 该机制的完整实例
- [[fast-weight-scaling-taxonomy]] — 「专家构造」维在分类法中的位置
- [[moba-attention]] — 刚性块专家的对照
- [[mixture-of-experts]] — 慢权重 MoE 与 fast-weight 专家的差别
- [[kv-cache-compression]] — 推理期逐 token KV 选择，属另一个问题

[^src-2602-01219-mita]: [[source-2602-01219-mita]]
