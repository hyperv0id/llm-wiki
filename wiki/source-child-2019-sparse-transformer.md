---
title: "Generating Long Sequences with Sparse Transformers"
type: source-summary
tags:
  - sparse-attention
  - efficient-transformer
  - autoregressive
  - long-sequence
  - OpenAI
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Generating Long Sequences with Sparse Transformers

作者为 Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever（OpenAI），arXiv 为 1904.10509v1（2019-04-23，正文 10 页，无附录），本地存档为 `raw/sparse-transformer-1904-10509.pdf`。

## 核心论点

自注意力的计算与显存随序列长度平方增长，长序列难以适用。论文将注意力矩阵按 $p=2$ 个头做稀疏分解（包含 strided 与 fixed 两种模式，[[sparse-transformer-strided-fixed|详见技术页]]），每头在每处位置只看 $O(\sqrt{n})$ 个历史位置，并保证任意输入-输出位置对经长度至多 $p+1$ 的路径连通，总代价降为 $O(n\sqrt{n})$（§4.2–4.3）[^src-child-2019-sparse-transformer]。该设计源自对 128 层 full-attention 网络的观察：CIFAR-10 上学到的注意力大多稀疏，早期层近似卷积，第 19–20 层自发拆为行、列两个注意力（Figure 2，§4.1）。

配套三项工程改动把模型推到数百层、数万步序列：pre-activation 残差块加 $1/\sqrt{2N}$ 初始化（§5.1–5.2）；反向传播时重算注意力与前馈激活以省显存，仅此即可在 16,384 长度上训练数百层 dense attention（§5.4）；block-sparse GPU kernel，上三角不计算、softmax 融合（§5.5）。

## 证据

密度建模，单位 bits per dim/byte（Table 1）：CIFAR-10 2.80（59M，strided，128 层；此前最优 PixelSNAIL 2.85）；Enwik8 0.99（95M，fixed，上下文 12,288，追平 277M Transformer-XL 的 0.99）；ImageNet 64×64 3.44（152M，strided；此前 SPN 150M 3.52）；12kHz 古典音乐 1.97（Table 1、Table 4）。同规模 dense 对照（Table 2）：Enwik8 上 fixed 0.99 对 dense 1.00，相对每步时间 0.55 对 1.31；strided 在文本失配（1.13），在 CIFAR-10 最低（2.80）。评估上下文从 6,144 加长到 12,160，Enwik8 误差单调降至 0.9908（Table 3）。音频上序列长度 ×4 需容量约 ÷$4\sqrt{4}=8$，百万步序列只能带 3M 参数（Table 4，§7.4）。

## 局限

实验仅在密度建模任务验证，包括图像字节（CIFAR-10、ImageNet-64）、文本字节（Enwik8）与 µ-law 音频，无机器翻译或摘要实验，全文无 BLEU 类指标。strided 依赖数据与 stride 对齐，文本需换 fixed（§4.3）。音频超过 65,536 步后样本质量因容量缩减而下降（§7.4）。dense 对照只有 Table 2 的 Enwik8 与 CIFAR-10 两处。

[^src-child-2019-sparse-transformer]: [[source-child-2019-sparse-transformer]]
