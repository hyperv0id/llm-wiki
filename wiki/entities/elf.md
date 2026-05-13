---
title: "ELF (Embedded Language Flows)"
type: entity
tags:
  - diffusion-language-model
  - flow-matching
  - mit
  - kaiming-he
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# ELF (Embedded Language Flows)

ELF (Embedded Language Flows) 是一种基于连续时间 [[flow-matching|Flow Matching]] 的[[diffusion-model|扩散]]语言模型，由 MIT 的 Kaiming He 团队提出[^src-elf]。

## 核心设计

ELF 的核心思想是：**在连续 embedding 空间中做 Flow Matching 去噪，只在最后一个时间步离散化到 token**。这一设计让它与之前的所有连续 DLM 和离散 DLM 都不同[^src-elf]。

### 技术要点

1. **连续 embedding 空间**：使用冻结的预训练 T5 encoder 将离散 token 映射到连续 embedding（512-dim），经瓶颈压缩至 128-dim，作为 Flow Matching 的操作空间[^src-elf]
2. **x-prediction**：网络预测干净 embedding x（而非 velocity v 或 noise ε），使得 denoising 和 decoding 的权重共享成为可能[^src-elf]
3. **共享权重 denoiser-decoder**：同一网络在训练时以 80% 概率做 denoising（MSE loss）、20% 概率做 decoding（CE loss），靠 mode token 区分[^src-elf]
4. **CFG 原生支持**：由于 ELF 在连续空间操作，classifier-free guidance 直接可用——这是离散 DLMs 做不到的[^src-elf]
5. **In-context conditioning**：将时间、CFG scale、mode 等条件信号作为 control token prepend 到序列中，而非 adaLN-Zero 加法融合[^src-elf]

## 模型变体

| 变体 | 参数量 | 配置 |
|------|--------|------|
| ELF-B | 105M | 默认配置 |
| ELF-M | 342M | 更大 |
| ELF-L | 652M | 最大 |

## 关键结果

- 无条件生成：Gen. PPL 24 @ 32步，击败离散 DLM（MDLM、Duo）和连续 DLM（FLM、LangFlow）[^src-elf]
- 训练效率：仅用 45B token，其他方法 500B+（10x 优势）[^src-elf]
- 翻译：WMT14 De-En BLEU 26.4（最佳）[^src-elf]
- 摘要：XSum ROUGE-1 36.0（最佳）[^src-elf]

## 相关方法

- [[flow-matching]] — ELF 的基础框架
- [[continuous-diffusion-language-model]] — ELF 所属的方法类别
- [[latent-diffusion-models|LDM]] — 潜在扩散，ELF 与之相似但不需要 decoder

[^src-elf]: [[source-elf-embedded-language-flows]]