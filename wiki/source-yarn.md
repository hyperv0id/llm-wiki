---
title: "YaRN: Efficient Context Window Extension of Large Language Models"
type: source-summary
tags:
  - transformer
  - rope
  - context-extension
  - interpolation
  - 2023
created: 2026-04-28
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# YaRN 论文摘要

## 核心论点

本文提出 **YaRN (Yet another RoPE extensioN method)**，用于高效扩展 RoPE 位置编码模型的上下文窗口[^src-yarn]。YaRN 整合 NTK-aware 插值、NTK-by-parts 插值与注意力温度缩放，仅用 <0.1% 原始预训练数据微调即可达到 SOTA 上下文扩展性能[^src-yarn]。

## 方法演进

YaRN 是以下方法的集大成者[^src-yarn]：

1. **Position Interpolation (PI)**：将所有 RoPE 维度等比例拉伸，但会丢失高频信息[^src-yarn]
2. **NTK-aware 插值**：通过改变 RoPE 的 base 参数，对高频少缩放、低频多缩放[^src-yarn]
3. **NTK-by-parts 插值**：根据波长与上下文长度的关系，对不同维度采用不同插值策略[^src-yarn]
4. **YaRN**：NTK-by-parts + 注意力 softmax 前的温度缩放[^src-yarn]
5. **Dynamic Scaling**：推理时动态调整缩放因子，无需微调即可扩展 2x 以上[^src-yarn]

## 主要贡献

1. **完整记录** NTK-aware、Dynamic NTK、NTK-by-parts 三项此前未正式发表的方法[^src-yarn]（§1）
2. **YaRN 方法**：结合 NTK-by-parts 和注意力温度缩放，训练收敛最快[^src-yarn]
3. **128k 上下文**：将 Llama 2 7B/13B 从 4k 扩展到 128k，性能损失极小[^src-yarn]
4. **零微调扩展**：Dynamic-YaRN 在无微调情况下可扩展 2x 以上上下文[^src-yarn]

## 实验结果

长序列语言建模中，Proof-pile 的滑动窗口 perplexity 低于 PI 与 NTK-aware，$s=32$ 的 13B 模型在 131072 处为 2.24[^src-yarn]（Table 1、Table 5、Table 8）。Passkey 检索中，128k 模型在全部上下文范围内准确率 $>99\%$[^src-yarn]。标准基准评测中，ARC-c/HellaSwag/TruthfulQA 相对基线的变动在 $\pm1.7$ 以内，MMLU 跌幅最大（13B $s=32$ 由 55.8 降到 51.9），$s=16$ 与 $s=32$ 之间平均下降 0.49%[^src-yarn]（Table 3、§4.4）。YaRN 训练收敛速度最快，400 步即可达到 PI 1000 步的效果[^src-yarn]，且与 Flash Attention 2 完全兼容，无额外运行时开销[^src-yarn]。

## 关键发现

插值策略由波长决定，$\lambda \ll L$ 的维度不插值（只承担相对位置），$\lambda \geq L$ 的维度必须插值以避免外推[^src-yarn]。$\sqrt{1/t} = 0.1\ln(s) + 1$ 温度缩放公式在 LLaMA 7B-65B 和 Llama 2 上均适用[^src-yarn]。迁移学习中，从 $s=16$ 的 checkpoint 继续训练 200 步即可扩展到 $s=32$[^src-yarn]。

## 局限性

方法需要针对不同模型族调整 $\alpha$、$\beta$[^src-yarn]。温度公式 Eq. 15 只在 LLaMA 7b/13b/33b/65b 上拟合，Llama 2 沿用而非重新拟合，且论文未给出熵与 $t$ 的定量关系[^src-yarn]。评测止于 131072，$s=32$ 模型只在 64k 数据上训练[^src-yarn]。

## 引用

[^src-yarn]: [[source-yarn]]