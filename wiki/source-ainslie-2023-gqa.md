---
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
type: source-summary
tags:
  - attention
  - kv-cache
  - efficient-inference
  - llm
  - multi-query
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai（Google Research），arXiv:2305.13245v3（2023-12-23），EMNLP 2023。

自回归解码的瓶颈在访存：每步须将全部 K/V 读入芯片。多查询注意力（MQA）将 K/V 头压为一份，代价是损失质量与训练稳定性；T5、LLaMA 等公开模型均为多头，仅为推理速度另训模型并不划算（§1）[^src-ainslie-2023-gqa]。

两项贡献。**uptraining**：将 MHA checkpoint 组内 K/V 投影矩阵取平均作为新 K/V 头，再沿用原配方继续预训练 $\alpha$ 比例的原预训练步数，$\alpha=0.05$ 约 600 TPUv3 chip-days（§2.1、§3.1）。**GQA**：query head 分为 $G$ 组，每组共享一份 K/V，GQA-1 即 MQA，GQA-$H$ 即 MHA；中间档质量接近 MHA，速度接近 MQA（§2.2）。

证据取自 T5.1.1 Large 与 XXL，任务涵盖五个摘要集、WMT 英德与 TriviaQA。Table 1 中 $\alpha=0.05$ 的 GQA-8-XXL 平均 47.1，MHA-XXL 47.2，uptrained MQA-XXL 46.6；每样本推理时间对应 0.28、1.51 与 0.24。Figure 3 给出四点的质量–时间前沿。Figure 4 的初始化消融中，mean pooling 优于单头与随机初始化；Figure 5 显示 MQA 与 GQA 均在 5% uptraining 后获得主要收益，10% 后边际递减；Figure 6 显示组数从 1 增至 8 仅增少量开销，以 $G=8$ 为折中。Appendix A 记录从零训练的 MQA 遇 loss spike 且长输入微调即发散，uptrained MQA 方差仍大，uptrained GQA 保持稳定。

验证仅在 encoder-decoder 上展开，任务限于翻译、摘要与问答；uptraining 依赖原始 checkpoint 与原预训练配方；受算力限制未对比从头训练的 GQA，未测 uptraining 相对从头训练的差距；摘要质量依赖 ROUGE，未覆盖全部质量维度（Limitations）。

## 相关页面

[[grouped-query-attention]] 涵盖机制、数据流、读取量与 uptraining 的全部细节；[[multi-query-attention]] 为 $G=1$ 的端点；[[kv-cache-compression]] 说明 GQA 改结构而压缩方法改缓存内容。

[^src-ainslie-2023-gqa]: [[source-ainslie-2023-gqa]]
