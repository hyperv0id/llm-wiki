---
title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)"
type: source-summary
tags:
  - mixture-of-experts
  - conditional-computation
  - language-modeling
  - machine-translation
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# source: Outrageously Large Neural Networks (Shazeer et al., 2017)

**归档**：`raw/shazeer-2017-sparse-moe-1701-06538.pdf`（arXiv:1701.06538v1，ICLR 2017，19 页）

Shazeer et al.（Google Brain）把条件计算做成通用组件 sparsely-gated MoE 层：$n$ 个同构单隐层 FFN expert 加门控网络，线性打分加幅值可学习的高斯噪声后取 top-$k$ 再 softmax，每样本只算 $k$ 个 expert，卷积式插在堆叠 LSTM 间逐时间步路由；为压制门控自增强地只选少数 expert，加 $CV^2$ 的 importance/load 双软约束，后者经噪声 CDF 把离散样本计数松弛成可微负载；训练为标准层同步数据并行、expert 单副本模型并行，expert batch 从 $kb/n$ 恢复到 $kbP/n$[^src-shazeer-2017-sparse-moe]。

论文报告：1B word 等算力下 4096 expert ppl 降低 24%，低算力档以最优模型 6% 运算量在 10 epochs 超过它（Table 1/7）；100B word 上 68B 参数达 28.9 对基线 47.0（−39%），131072 expert 回落至 29.2（Table 8）；WMT'14 En→Fr/En→De 为 40.56/26.03 BLEU，比无 RL 的 GNMT 高 1.34/1.12（Table 2/3）；多语 12 对中 11 对优于 GNMT-Multi（dev ppl 3.35 对 4.14，Table 5），仅 En→Ko 因稀有语言对过采样过训；消融仅覆盖均衡损失权重（Table 6）：任一损失即恢复质量，最大负载比从 17.8 降至 1.1–1.5[^src-shazeer-2017-sparse-moe]。

实验均基于 LSTM 架构，收益依赖大数据——1B word 上 MoE 层容量过 1B 参数后递减——且最大模型效率仅 0.30 TFLOPS/GPU；论文未提供 $k$ 扫描与噪声开关消融[^src-shazeer-2017-sparse-moe]。

机制与数据流见 [[sparse-gated-moe-layer]]。

[^src-shazeer-2017-sparse-moe]: [[source-shazeer-2017-sparse-moe]]
