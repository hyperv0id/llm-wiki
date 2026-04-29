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
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# YaRN 论文摘要

## 核心论点

本文提出 **YaRN (Yet another RoPE extensioN method)**，一种高效扩展 RoPE 位置编码模型上下文窗口的方法[^src-yarn]。YaRN 整合了 NTK-aware 插值、NTK-by-parts 插值和注意力温度缩放三项技术，在仅使用 <0.1% 原始预训练数据微调后即可达到 SOTA 上下文扩展性能[^src-yarn]。

## 方法演进

YaRN 是以下方法的集大成者[^src-yarn]：

1. **Position Interpolation (PI)**：将所有 RoPE 维度等比例拉伸，但会丢失高频信息[^src-yarn]
2. **NTK-aware 插值**：通过改变 RoPE 的 base 参数，对高频少缩放、低频多缩放[^src-yarn]
3. **NTK-by-parts 插值**：根据波长与上下文长度的关系，对不同维度采用不同插值策略[^src-yarn]
4. **YaRN**：NTK-by-parts + 注意力 softmax 前的温度缩放[^src-yarn]
5. **Dynamic Scaling**：推理时动态调整缩放因子，无需微调即可扩展 2x 以上[^src-yarn]

## 主要贡献

1. **首次完整记录** NTK-aware、Dynamic NTK、NTK-by-parts 等此前未正式发表的方法[^src-yarn]
2. **YaRN 方法**：结合 NTK-by-parts 和注意力温度缩放，训练收敛最快[^src-yarn]
3. **128k 上下文**：将 Llama 2 7B/13B 从 4k 扩展到 128k，性能损失极小[^src-yarn]
4. **零微调扩展**：Dynamic-YaRN 在无微调情况下可扩展 2x 以上上下文[^src-yarn]

## 实验结果

- **长序列语言建模**：YaRN 在 Proof-pile 和 GovReport 上 perplexity 优于 PI 和 NTK-aware[^src-yarn]
- **Passkey 检索**：128k 模型在全部上下文范围内准确率 >99%[^src-yarn]
- **标准基准**：ARC-c、HellaSwag、MMLU、TruthfulQA 上性能损失极小（平均 <0.5%）[^src-yarn]
- **训练效率**：YaRN 收敛速度最快，400 步即可达到 PI 1000 步的效果[^src-yarn]
- **兼容性**：与 Flash Attention 2 完全兼容，无额外运行时开销[^src-yarn]

## 关键发现

- **波长决定插值策略**：λ ≪ L 的维度不插值（保留相对位置），λ ≥ L 的维度必须插值（避免外推）[^src-yarn]
- **温度缩放的普适性**：$\sqrt{1/t} = 0.1\ln(s) + 1$ 公式在 LLaMA 7B-65B 和 Llama 2 上均适用[^src-yarn]
- **迁移学习**：从 s=16 的 checkpoint 继续训练 200 步即可扩展到 s=32[^src-yarn]

## 局限性

- 需要针对不同模型族调整 α、β 参数[^src-yarn]
- 温度缩放公式的经验拟合可能不适用于所有架构[^src-yarn]

## 引用

[^src-yarn]: [[source-yarn]]