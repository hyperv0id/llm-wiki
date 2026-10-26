---
title: "source-gu-dao-2023-mamba"
type: source-summary
tags:
  - source-summary
  - mamba
  - state-space-model
  - linear-complexity
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# source-gu-dao-2023-mamba

Mamba: Linear-Time Sequence Modeling with Selective State Spaces。Albert Gu（CMU）、Tri Dao（Princeton），本库归档 arXiv:2312.00752v2（2024-05-31，36 页，§1–§6 与附录 A–E）[^src-gu-dao-2023-mamba]。

**问题与论点**：Transformer 无法在有限窗口外建模，计算量随窗口长度二次增长；线性注意力、门控卷积/RNN、结构化 SSM 等次二次架构在语言等信息密集的离散模态上不敌 attention。作者将该弱点归结为无法依输入选择数据——不能按当前 token 决定记住或忽略什么[^src-gu-dao-2023-mamba]。

**方法**：三件事。一是选择机制：把 SSM 参数 $\Delta, B, C$ 设为输入函数（S4→S6），沿序列选择性传播或遗忘信息；二是硬件感知算法：选择使时不变性与卷积等价失效，改在循环模式下以 kernel 融合 + 并行 scan + 重计算把 $(B,L,D,N)$ 扩展态留在 SRAM，IO 减少 $O(N)$ 倍；三是简化架构：H3 块与 MLP 块合并为同构堆叠的 Mamba block，无 attention、无独立 MLP[^src-gu-dao-2023-mamba]。

**证据**：Selective Copying 上 S6 层 97.0–99.8 对 S4 层 18.3–57.0（Table 1）；Induction Heads 外推至训练长度的 4000 倍（1M），其余方法不超过 2 倍（Table 2、Table 11）。语言建模 125M–1.3B Chinchilla 缩放下首个匹配 Transformer++ 的无注意力架构（Figure 4）；300B token 下 Mamba-2.8B 七任务平均 63.3 对 Pythia-2.8B 59.1、超 Pythia-6.9B 61.7（Table 3）。DNA HG38（约 4.5B token）：在约 40M 参数处以少 3–4× 参数匹配 Transformer++/HyenaDNA，困惑度随上下文增至 1M 单调改善而 HyenaDNA 变差（Figure 5）；大猿分类 1M 长度 81.31% 对 HyenaDNA 54.87%（Table 13）。音频 SC09 生成 FID 0.94（6.1M）/0.67（24.3M）对 SaShiMi 1.99（Table 4）；推理吞吐 4–5×（§4.5）[^src-gu-dao-2023-mamba]。

**局限**：缩放仅验证至约 3B，未评估 7B+；微调/ICL/RLHF 等下游能力未探索；选择机制损伤连续均匀信号，长音频波形上 LTI（S4）更优（Figure 10）[^src-gu-dao-2023-mamba]。

机制主体见 [[selective-ssm]]，实体页见 [[mamba]]。

[^src-gu-dao-2023-mamba]: [[source-gu-dao-2023-mamba]]
