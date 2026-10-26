---
title: "Universal Transformers (Dehghani et al., ICLR 2019)"
type: source-summary
tags:
  - architecture
  - recurrence
  - adaptive-computation
  - weight-tying
  - language-model
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Universal Transformers (Dehghani et al., ICLR 2019)

Mostafa Dehghani、Stephan Gouws、Oriol Vinyals、Jakob Uszkoreit、Łukasz Kaiser（University of Amsterdam / DeepMind / Google Brain），*Universal Transformers*，ICLR 2019，arXiv:1807.03819v3（2019-03-05）；PDF：`raw/universal-transformers-1807-03819.pdf`[^src-dehghani-2019-universal-transformers]。

## 问题

标准 Transformer 采用固定层数与逐层独立权重的堆叠，缺少 RNN 对迭代/递归变换的归纳偏置；实验显示其在超出训练长度的任务（copy、简单逻辑推理）上泛化差，而 RNN 类模型能顺利处理这类任务（§1）[^src-dehghani-2019-universal-transformers]。

## 方法

将 Transformer 改为沿**深度**循环的并行自注意力 RNN：同一组自注意力与 transition 权重跨所有位置和循环步共享，每步先算跨位置自注意力，再逐位置经 transition（separable 卷积或两层仿射夹 ReLU），输入每步叠加位置与步数的正弦坐标编码（§2.1，式 4–7）[^src-dehghani-2019-universal-transformers]。辅以基于 ACT（Graves, 2016）的逐位置动态停机：各位置每步预测标量停机概率，累积超阈值即停并复制状态，直至全停或达最大步数（§2.2、附录 C）[^src-dehghani-2019-universal-transformers]。循环步取 1 即标准 Transformer；在内存充足且步数为输入长度函数的条件下，证明 UT 计算通用，并将 Neural GPU 归约为 UT 的特例（§4、附录 B）[^src-dehghani-2019-universal-transformers]。

## 证据

bAbI 平均错误率 0.21/0.29（10K single/joint）、4.55/7.78（1K），20 任务零失败，均优于 Transformer 的 15.2–26.8（Table 1）[^src-dehghani-2019-universal-transformers]。LAMBADA 动态停机版 test PPL 142、RC 准确率 0.5625，达新最好值（Table 3）[^src-dehghani-2019-universal-transformers]。WMT14 En-De 达 28.9 BLEU，比同参数量 Transformer base 高 0.9（Table 7）[^src-dehghani-2019-universal-transformers]。算法任务（train 40 → test 400）与 Learning to Execute 上大幅超过 LSTM 与 Transformer（Tables 4–6）[^src-dehghani-2019-universal-transformers]。停机步数随任务难度上升，支撑事实 1/2/3 个时平均 ponder 时间为 $2.3\pm0.8$ / $3.1\pm1.1$ / $3.8\pm2.2$（§3.1）[^src-dehghani-2019-universal-transformers]。

## 局限

ACT 在机器翻译上轻度降低结果（§1）。标准 Transformer 在 bAbI 上换超参与规模均过拟合（§3.1 脚注 2），且 1K 联合训练下任务 17/19 仍失败（附录 E）。算法任务长度外推改善但不完美，Neural GPU 的满分依赖专门 curriculum（§3.4，Table 4）。此外，图灵完备依赖有限精度之外的假设（附录 B）[^src-dehghani-2019-universal-transformers]。

[^src-dehghani-2019-universal-transformers]: [[source-dehghani-2019-universal-transformers]]
