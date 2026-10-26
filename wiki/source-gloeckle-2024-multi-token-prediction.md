---
title: "Better & Faster Large Language Models via Multi-token Prediction"
type: source-summary
tags:
  - llm
  - pretraining
  - multi-token-prediction
  - speculative-decoding
  - code-generation
  - sample-efficiency
created: 2026-07-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Better & Faster Large Language Models via Multi-token Prediction

**Authors**: Fabian Gloeckle\*, Badr Youbi Idrissi\*, Baptiste Rozière, David Lopez-Paz⁺, Gabriel Synnaeve⁺ (FAIR at Meta / CERMICS École des Ponts / LISN Université Paris-Saclay). ICML 2024 / arXiv:2404.19737.

## 核心问题

GPT/Llama 类 LLM 几乎一律用 **next-token prediction + teacher forcing** 预训练。但这种方式样本效率较低：模型易贴合局部模式、忽视“硬选择点”，且训练（教师强制）与推理（自回归）分布不一致。其目标是在不增加训练时间与峰值显存的前提下，用多 token 预测提升样本效率与推理速度[^src-gloeckle-2024-multi-token-prediction]。

## 核心方法

在每个训练位置，用 **共享 trunk + n 个独立输出头** 并行预测后续 n 个 token；推理默认只用 next-token 头，其余头可用于 self-speculative decoding[^src-gloeckle-2024-multi-token-prediction]。

- **损失**（条件独立分解，式 (2)）：
  $$L^n = -\sum_t \sum_{i=1}^{n} \log P_\theta (x_{t+i}\mid z_{t:1})\,P_\theta (z_{t:1}\mid x_{t:1})$$
共享 trunk $f_s$ 产出表示 $z$，接 n 个通常各一层 transformer 的独立 head $f_{h_i}$ 与共享 unembedding $f_u$；公平对比时，增加 n−1 个 head 层便从 trunk 扣减 n−1 层以匹配总参数量。trunk 前向一次后逐 head 串行 forward/backward，只在 trunk 累积梯度，峰值显存从 $O(nV+d)$ 降到 $O(V+d)$，训练时间几乎无开销。推理时标准自回归用 $i=1$ 头，亦可用 blockwise parallel / Medusa 式 self-speculative decoding 配合多 head 加速。

## 主要结果

1. **随规模增强**：代码语料上 300M–13B，小模型多 token 常弱于 baseline，大模型反超。13B 相对 next-token 约多解 12% HumanEval、17% MBPP 问题（摘要；逐尺寸数据见表 S7）[^src-gloeckle-2024-multi-token-prediction]。
2. **推理加速**：7B 的 4-token 模型 self-speculative 约 **3×**（code 平均接受 ~2.5/3；text ~2.7×）；8-byte 头可达 **~6.4×**（§3.2、表 S2/S3）[^src-gloeckle-2024-multi-token-prediction]。
3. **Byte-level**：7B 字节模型上 8-byte 预测相对 next-byte：MBPP pass@1 +67%、HumanEval pass@1 +20%；有望抵消更长字节序列的推理成本（§3.3、表 1）[^src-gloeckle-2024-multi-token-prediction]。
4. **最优 n**：7B / 200B code tokens / 32k vocab 上 **n=4** 在 HumanEval/MBPP 最稳；APPS/Intro 上 n=6 更好——最优窗口依赖数据分布（§3.4、表 1）[^src-gloeckle-2024-multi-token-prediction]。
5. **多 epoch**：1T tokens（4 epoch）仍保留增益（如 MBPP pass@1 +2.4%），幅度缩小（§3.5、表 1）[^src-gloeckle-2024-multi-token-prediction]。
6. **微调**：CodeContests 上，4-token **预训练** 后用 $n'{=}1$ 或 $n'{=}4$ 微调都强于 next-token 预训练；**多 token 预训练 + next-token 微调** 整体最佳（§3.6、图 4）[^src-gloeckle-2024-multi-token-prediction]。
7. **自然语言**：7B 上 multiple-choice/likelihood 基准几乎不升甚至 n=4 略退；**摘要** ROUGE 有稳定提升；GSM8K 在 200B tokens 时 n=2 更好，500B 后次序翻转（§3.7、图 5/6、图 S13）[^src-gloeckle-2024-multi-token-prediction]。
8. **合成任务**：多 token 促进小模型 **induction heads** 形成；多项式算术上 OOD 泛化提升，效果常大于把模型从 30M 扩到 100M（§4.1/4.2、图 7/8/S16）[^src-gloeckle-2024-multi-token-prediction]。

## 机制直觉（作者推测）

在 lookahead 视角下，难预测的关键转移及其后果在 n-token loss 中获得约 $\frac{n(n+1)}{2}$ 权重，无关紧要转移约 n 倍，更强调语义转折点（§5.1、附录 L.3）[^src-gloeckle-2024-multi-token-prediction]。信息论分析中，2-token 目标相对放大 $I(X;Y)$（相对互信息权重为 2），迫使表示编码对后续文本有约束力的决策，而非纯局部风格变体（§5.2、附录 L.2）[^src-gloeckle-2024-multi-token-prediction]。对比 scheduled sampling，离散文本上交错真值与模型 token 易产生不连贯序列，多 token 预测则可保持并行训练且不破坏输入（附录 L.1）[^src-gloeckle-2024-multi-token-prediction]。

## 局限与意义

增益集中在**生成/推理/代码**，选择题基准未见提升，小模型可能变差[^src-gloeckle-2024-multi-token-prediction]。n 与词表大小需依数据选取；在已 next-token 预训练的 Llama-2 上直接加多 token 微调改善有限（表 S6）[^src-gloeckle-2024-multi-token-prediction]。其贡献是将 multi-token 从 ProphetNet/Medusa 等“小规模或仅加速”叙事，推至参数与时间公平的**大规模预训练主损失**，并给出可落地的显存实现[^src-gloeckle-2024-multi-token-prediction]。

## 相关页面

[[multi-token-prediction]] 为技术主页，[[self-speculative-decoding]] 对应多头自推测解码，[[sparse-teacher-forcing]] 是另一类在 DSR 场景下缓解 teacher-forcing 与自回归分布错配的训练法。

## 引用

[^src-gloeckle-2024-multi-token-prediction]: [[source-gloeckle-2024-multi-token-prediction]]
