---
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
type: source-summary
tags:
  - mixture-of-experts
  - sparse-activation
  - routing
  - scaling
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**Authors**: William Fedus*, Barret Zoph*, Noam Shazeer（Google, Mountain View）。JMLR 23 (2022) 1–40；arXiv:2101.03961v3（2022-06-16）。

## 核心问题

MoE 虽在机器翻译上有效，但广泛应用受制于三大阻碍：实现复杂度、通信成本与训练不稳定[^src-switch-transformer-2101-03961]。论文沿缩放定律开辟第四轴——在每 token FLOPs 不变的前提下放大参数量，以稀疏激活的 expert 层实现[^src-switch-transformer-2101-03961]。

## 方法

对 MoE 连做三项减法：路由从 top-k 简化为 top-1（Switch 层），使 expert 容量至少减半、派发通信简化[^src-switch-transformer-2101-03961]；负载均衡合并为单一乘性辅助损失 $\alpha \cdot N \sum_e f_e P_e$（$\alpha = 10^{-2}$），取代 Shazeer 2017 的加性双损失[^src-switch-transformer-2101-03961]；bfloat16 训练下仅在 router 函数体内保留 float32（selective precision），通信张量仍为 bfloat16[^src-switch-transformer-2101-03961]。配套方案：初始化尺度缩 10 倍以稳定 223M 到 1.6T 全谱系，微调时仅 expert 内部 dropout 提至 0.4，稀疏模型可蒸馏回 dense（非 expert 权重初始化 + 0.25/0.75 软硬损失混合）[^src-switch-transformer-2101-03961]。

## 证据

预训练：FLOP-matched 下 Switch-Base 达 T5-Base 质量只需 1/7 墙钟（64 expert）[^src-switch-transformer-2101-03961]；expert 数 2→256 一致改善[^src-switch-transformer-2101-03961]；Switch-C（1571B 参数、2048 expert）同算力比 T5-XXL 快 4 倍，500k 步 Neg. Log Perp. -1.043[^src-switch-transformer-2101-03961]。微调：SuperGLUE 79.5 vs T5-Base 75.1（FLOP-matched）[^src-switch-transformer-2101-03961]；ANLI 65.7 对此前最好 49.4[^src-switch-transformer-2101-03961]；闭卷 QA 三项超 T5-XXL[^src-switch-transformer-2101-03961]。多语言：101 语言全部改善，平均 5 倍 step 加速[^src-switch-transformer-2101-03961]。蒸馏：97–99% 压缩下保留约 30% 增益[^src-switch-transformer-2101-03961]。

## 局限（论文自述）

Switch-XXL（FLOPs 最大档）间歇不稳定[^src-switch-transformer-2101-03961]；最大档上游困惑度优势未完全转化为推理型下游（Switch-C SQuAD 87.7 低于 Switch-XXL 89.6，论文称 poorly understood）[^src-switch-transformer-2101-03961]；attention 内 expert 层 bfloat16 发散[^src-switch-transformer-2101-03961]；蒸馏不能完整保留质量[^src-switch-transformer-2101-03961]。

[^src-switch-transformer-2101-03961]: [[fedus-2022-switch-transformer]]
