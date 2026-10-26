---
title: "GLU Variants Improve Transformer"
type: source-summary
tags:
  - feed-forward
  - activation-function
  - gated-linear-unit
  - transformer
  - arxiv-2020
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# GLU Variants Improve Transformer

**作者**：Noam Shazeer（Google）；**版本**：arXiv:2002.05202v1 [cs.LG]，2020-02-12；5 页，无附录；PDF 未标注会议或期刊（预印本）[^src-shazeer-2020-glu-variants]；**归档**：`raw/glu-variants-improve-transformer-2002-05202.pdf`

## 核心贡献

论文将 GLU 从「sigmoid 门控」推广为一族门控激活，置入 Transformer 逐位置 FFN 做等参数对照。GLU 层是两路线性投影的逐元素积，其中一路先经 sigmoid（式 4）；将 sigmoid 替换为 ReLU、GELU、Swish 或恒等得到式 5 变体，再接第三个矩阵 $W_2$ 降回 $d_{\text{model}}$ 即为 FFN 变体（式 6）[^src-shazeer-2020-glu-variants]。

三矩阵相比双矩阵，为保持参数量与算力，内层宽度按 $3/2$ 折减，T5-base 的 $d_{ff}$ 由 3072 折至 2048（式 6 下方、§3.1）[^src-shazeer-2020-glu-variants]。折算后内层总通道 $2d'_{ff}=\tfrac{4}{3}d_{ff}$ 宽于基准，但经激活的只有 $d'_{ff}$ 一路，另一路是不带非线性的内容支路；Bilinear 则取消门控激活，将「门控非线性」与「两路相乘」两个因素分开[^src-shazeer-2020-glu-variants][INFERENCE]。

## 实验

T5-base（编码器/解码器各 12 层、$d_{\text{model}}=768$、$h=12$、$d_k=d_v=64$）在 C4 上完成 524,288 步 span-filling 预训练，batch 128，Adafactor，不加 dropout；另训 4 个 65,536 步模型测方差[^src-shazeer-2020-glu-variants]。heldout log-perplexity（表 1）：GEGLU 1.633、SwiGLU 1.636 最低，ReLU 基线 1.677，五个门控变体均低于 ReLU、GELU、Swish[^src-shazeer-2020-glu-variants]。微调 131,072 步后：GLUE 均分 ReGLU 84.67、SwiGLU 84.36 对 ReLU 83.80；SuperGLUE SwiGLU 74.56 对 72.76；SQuAD F1 ReGLU 91.18 对 90.87（表 2–4）[^src-shazeer-2020-glu-variants]。GLU 在 SQuAD 两项均低于 ReLU。

## 边界

论文未解释各变体生效的原因。实验基于 T5-base 文本迁移设置：长程预训练为单次运行（方差仅由 65,536 步档给出），各模型微调一次并取 dev 最佳 checkpoint；微调结果存在噪声，表内标准差行取自 Raffel et al. 的多次运行、未覆盖本表各变体[^src-shazeer-2020-glu-variants]。机制拆解与数据流见 [[swiglu]]。

[^src-shazeer-2020-glu-variants]: [[source-shazeer-2020-glu-variants]]
