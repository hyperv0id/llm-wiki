---
title: "Fast-Weight Scaling 分类法：高效注意力的五维坐标"
type: analysis
tags:
  - attention
  - efficient-attention
  - taxonomy
  - mixture-of-experts
  - fast-weights
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Fast-Weight Scaling 分类法

MiTA 论文（arXiv:2602.01219，§3.1、Table 1）基于 [[fast-weight-mlp-view-of-attention|fast-weight 视角]]，将高效注意力划入五个维度：缩放策略、专家数量、专家类型、专家构造、路由拓扑[^src-2602-01219-mita]。

## 五个维度

| 维度 | 论文定义 | 取值 |
|---|---|---|
| 缩放策略 | 压缩成轻量模块，还是把 query 路由到 $N$ 宽 MLP 的一部分 | Compression / Routing / 两者兼用 |
| 专家数量 | 构造多少个 fast-weight 专家 | 1（共享）/ $m$ / $N/n$ / $N$ |
| 专家类型 | 每个专家取什么模块 | 线性层 / MLP / 任意模块 |
| 专家构造 | 专家如何由 key-value 对形成 | 依 query / 依 KV / 依两者；内容相关（形变）vs 位置驱动 |
| 路由拓扑 | query 到专家的指派模式 | all-to-one / $N$-to-one / $N$ 次 one-to-one / $N$-to-$N/n$ / $N$-to-$m$ |

论文将专家数量与专家类型合并为一个复合维度，正文未展开说明原因[^src-2602-01219-mita]。

## 方法映射（Table 1）

| 方法 | 缩放策略 | 专家类型与数量 | 专家构造 | 路由拓扑 |
|---|---|---|---|---|
| Linear attention (2020) | Compression | 1 个共享线性层 | 核化 | all-to-one |
| MHLA (2026) | 压缩 | $m$ 个线性层 | 局部先验 + 核化 | $m$ 次 $N/m$-to-one |
| Linformer (2020) / PVT (2021) | Compression | 1 个共享 MLP | 可学习投影 | all-to-one |
| Agent attention (2024) | Compression | 1 个共享 MLP | landmark testing | all-to-one |
| TTT (2025) / ViT3 | Compression | 1 个共享模块 | test-time training | all-to-one |
| Deformable DETR | Routing | $N$ 个 MLP | 偏移预测 | $N$ 次 one-to-one |
| DAT (2022) | Routing | 1 个共享 MLP | 依 query | $N$-to-one |
| MoBA (2025) | Routing | $N/n$ 个 MLP | 局部性先验 | $N$-to-$N/n$ |
| Spark attention (2025) / DSA (2025) | Routing | $N$ 个 MLP | 轻量 skim（部分特征 / FP8） | $N$ 次 one-to-one |
| **MiTA** | **Compression & Routing** | **$m$ 个 MLP** | **landmark testing** | **$N$-to-$m$** |

表依原表顺序排列。转换丢失的 PVT、ViT3、DSA 与 DAT 策略列按上一行合并单元格补齐[^src-2602-01219-mita]。

## 这套坐标给出的读法

- **压缩与路由对应不同的失效模式**。论文指出：单纯压缩会损失对原始 KV 的精确访问，单纯路由则缺少全局上下文概括（§1）[^src-2602-01219-mita]。MiTA 与 NSA 为表中兼具两者的少数方法。
- **专家数量与硬件友好性挂钩**。逐 query 生成 $N$ 个 top-$k$ 专家（Spark、DSA）时每 query 的专家形状都不同；固定 $m$ 个专家则可以块式批量计算，代价是多一次 $N$-to-$m$ 路由[^src-2602-01219-mita]。
- **「专家构造」界定形变与刚性**。MoBA 依据位置划分块，MiTA 依据 $Q^{\top}\tilde Q$ 的 top-$k$ 聚合 KV，详见 [[deformable-fast-weight-experts]]。

## 与 wiki 中既有分类的关系

wiki 已有三套局部划分，覆盖范围与这套五维不同，引用时按范围区分：

| 既有分类 | 覆盖 | 与五维的关系 |
|---|---|---|
| [[linear-attention-unified-framework]] | Mamba ↔ 线性注意力的公式级对应 | 五维只把它放在「压缩 + 1 个共享线性层 + all-to-one」一格 |
| [[cbsa]] / [[representative-token-extraction]] | 以代表性 token 取法实例化 softmax / linear / channel / agent | 对应五维的「专家构造」一维，但给的是等价公式而非坐标 |
| [[patchstg-sparse-attention-form]] | 线性化 / 改变交互拓扑 / 结构化稀疏三路线 | 与五维中的「缩放策略 + 路由拓扑」部分重叠，且只覆盖掩码与块结构 |

## 分类法自身的边界

- 分类维度是**结构**，不含评测口径：同一格里既有 LLM 长上下文方法（MoBA、NSA、DSA），也有视觉方法（Agent attention、PVT），跨域可比性不由表保证。
- 分类未量化**压缩率与选择粒度的开销**：$m$ 与 $k$ 的取值及 gather 访存开销未纳入坐标；论文在 §3.2 中将 gather 列为潜在显存带宽瓶颈[^src-2602-01219-mita]。
- Table 1 的 NSA 行转换后仅保留 «Compression & Routing»，其余三维缺失，无法从本地文本恢复其完整定位[^src-2602-01219-mita]。
- Table 1 将 Deformable DETR 标注为 2017（对应 Dai et al. 的 deformable convolution），正文 §2.3 则引为 Zhu et al. 2021，两处年份不一致[^src-2602-01219-mita]。
- 论文自述该分类法「by no means complete」（§1）[^src-2602-01219-mita]。

## 「多出来的那一支是否付账」

该分类法未提供增益幅度评估。可用的直接对比为 MiTA 与同族**纯压缩**方法 Agent attention：ImageNet-1K 从头训练（Table 2）MiTA-ViT-T 72.9 vs Agent-ViT-T$^{25}$ 68.9（+4.0）；ImageNet-21K 微调（Table 3）MiTA-ViT$^{25+25}$ 75.6/80.9/82.8/85.5 vs Agent-ViT$^{64}$ 74.7/79.9/81.5/83.8（+0.9/+1.0/+1.3/+1.7）；LRA（Table 5）平均 59.26 vs 58.40（+0.86）[^src-2602-01219-mita]。上述差值同时包含路由与形变专家两项改动，论文未作消融拆分，且未与块路由方法（MoBA、NSA）进行横向对比。

## 相关页面

- [[mita-attention]] — 表中最后一行的机制与证据
- [[fast-weight-mlp-view-of-attention]] — 该表所依赖的形式化
- [[deformable-fast-weight-experts]] — 专家构造维的展开
- [[moba-attention]] · [[mixture-of-experts]] — 路由支的代表机制与 MoE 原则
- [[cbsa]] · [[representative-token-extraction]] · [[linear-attention-unified-framework]] — 局部统一框架

[^src-2602-01219-mita]: [[source-2602-01219-mita]]
