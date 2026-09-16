---
title: "Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers"
type: source-summary
tags:
  - llm
  - long-context
  - reasoning
  - textual-state
  - arxiv
created: 2026-09-13
last_updated: 2026-09-13
source_count: 1
confidence: medium
status: active
---

# Source: Trace as State

## 著录与原始材料

- **作者与署名单位**：Xu Zou（Z.ai）、Jie Tang（Tsinghua University）。[^src-trace-as-state]
- **版本**：arXiv:2609.02702v1，2026-09-02；arXiv 著录为 preprint，不据此添加会议接收信息。[^src-trace-as-state]
- **原文**：[arXiv 著录](https://arxiv.org/abs/2609.02702v1) · [完整 HTML](https://arxiv.org/html/2609.02702v1) · [21 页 PDF](https://arxiv.org/pdf/2609.02702v1) · [TeX Source](https://arxiv.org/src/2609.02702v1)。
- **本地原文**：[[downloads/trace-as-state.pdf]]；TeX 包 `downloads/trace-as-state.tar.gz`；完整 `pdftotext -layout` 抽取为 `raw/trace-as-state.txt`（2026-09-15 由 downloads/ 迁入）。
- **许可与图版**：arXiv 声明 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本次只将原图 PDF 转为 PNG，未改图内内容。[^src-trace-as-state]
- **来源计数**：PDF、HTML、TeX 是同一篇论文的不同载体，合计一个来源；脚注 slug 为 `trace-as-state`。
- **归档边界**：PDF 与 TeX 包留存 `downloads/`；`pdftotext` 抽取文本自 2026-09-15 起存于 `raw/trace-as-state.txt`。本仓库不声称存在 `raw/trace-as-state.pdf`。
- **阅读范围**：正文 §1–5、Limitations、References、Appendix A–F；图版来自 TeX 包的原始 PDF。
- **PDF SHA256**：`85f79e1407f6d391f9a36907d5e8031f0ce84db9c094335095f69c3a9c887e1c`。
- **TeX SHA256**：`bfc04fe5b8c47a0d77d904dc285ee43ce32d843173cfa7ec4e9e64cb2aa8cb27`。

## 摘要

论文研究长上下文中的顺序错配：任务状态可能在读完材料后才形成，却不能改变同一遍的早期表示。[^src-trace-as-state]作者构造确定性、精确、单遍任务，证明条件先给与后给可有指数级工作内存差距；该定理只是方法动机，不是现实模型的显存界。[^src-trace-as-state]

方法把同题首轮推理轨迹前置于原文，再启动新的因果处理。[^src-trace-as-state]主实验复用五条轨迹，每条最多前五万字符，问题始终置末；对照使用同一轨迹但放在原文之后，单独返回的首轮答案不纳入。[^src-trace-as-state]

三模型在图遍历、历史对话绑定与长小说理解中，前置在27个指标格中的26个均值更高。[^src-trace-as-state]GraphWalks Parents的EM，DeepSeek从首轮29.2、后置43.0升至前置81.8；GLM-5.2从66.4、83.2升至100.0。[^src-trace-as-state]

附录24个配对差值中，20个未校正95%区间完全大于零；小说仅20题，不能把五次重复当独立样本。[^src-trace-as-state]作者指出该实现依赖可用的状态接口，增加重读成本、可能减少缓存复用，且未评估多轮agent。[^src-trace-as-state]轨迹可有错误，理论也未证明其忠实反映内部状态。[^src-trace-as-state]

## 深入阅读

- [[trace-as-state]] — 主入口：问题、方法、核心结果与范围。
- [[conditional-state-update]] — §3.1 / Appendix A 的完整证明与限定。
- [[trace-as-state-evidence]] — 主表、消融、配对区间与难度诊断。
- [[trace-as-state-reproduction]] — 提示、评分、接口版本、token 与公开材料。
- [[trace-as-state-positioning]] — 四篇关键相关论文的原始来源对照。

[^src-trace-as-state]: [[source-trace-as-state]]
