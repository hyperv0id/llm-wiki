---
title: "ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning (arXiv 2607.02509)"
type: source-summary
tags:
  - long-context
  - evidence-replay
  - training-free
  - inference-method
  - associative-memory
created: 2026-09-13
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# ReContext（Recursive Evidence Replay, arXiv:2607.02509）

## 摘要
ReContext 由 Yanjun Zhao 等 9 人提出，是 2026-07-02 的预印本。[^src-recontext]

论文关注已有上下文中的证据利用：材料虽然可访问，模型却未必在回答时使用它；作者提出免训练的递归证据回放。[^src-recontext]

方法用问题末尾 8 个线索位置对上下文的内部注意力相关性，经选定层头与衰减聚合取前 K 个位置，映射回原文句子；递归 R 轮维护有序证据池，每轮重打分并追加新片段。[^src-recontext]最终输入仍保留“完整上下文＋证据池＋问题”，不直接修改 attention logits，并复用 KV cache。[^src-recontext]

作者以联想记忆解释回放；定理 1 的逐步改进结论限于附录 E.1 的理想化设定，包括单位正交嵌入、答案对应某个上下文词元、问题对答案相关性严格最高及额外数值条件，不能当作真实模型的一般保证。[^src-recontext]

八个 128K 数据集、Qwen3-4B/8B 与 Llama3-8B 的实验中，作者报告相对 Vanilla 平均 accuracy 从 0.24 到 0.30（报告的相对增益为 24.6%），并在三个骨干上取得最佳平均排名。[^src-recontext]

作者自述局限是需要内部相关性信号、不能直接适配不暴露该信号的闭源 API，且读—回放增加延迟。[^src-recontext]
## 来源与阅读范围

- arXiv 摘要页：https://arxiv.org/abs/2607.02509
- PDF：https://arxiv.org/pdf/2607.02509 （v1，2026-07-02）
- 官方代码：https://github.com/Yanjun-Zhao/ReContext
- 本地锁定版本：`downloads/recontext.pdf`（PDF 留在 downloads/；pdftotext 抽取文本在 `raw/recontext.txt`）
- 阅读范围：pdftotext 全文提取后，完整阅读摘要、§1 引言、§2 相关工作、§3 方法（证据选择、物化与回放、递归选择、理论分析）、§4 数据集与主结果表、§Limitations、附录 E.1/E.2 理论设定与证明开头；其余结果表与附录为浏览级。

## 与主论文的关系

[[trace-as-state]] §2 转述 ReContext "recursively builds and replays an evidence pool for the current query"，与原作机制一致；其回放位置在上下文之后 [INFERENCE]，与前置 trace 的位置干预方向相反，两者无 head-to-head（见 [[trace-as-state-positioning]]）。[^src-trace-as-state]

[^src-recontext]: [[source-recontext]]
[^src-trace-as-state]: [[source-trace-as-state]]
