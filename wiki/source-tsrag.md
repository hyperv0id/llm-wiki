---
title: "TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecasters"
type: source-summary
tags:
  - time-series
  - forecasting
  - mixture-of-experts
  - retrieval-augmented-generation
  - foundation-models
  - zero-shot
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecasters

**Authors**: 缓存页眉标注 Anonymous Authors（"Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute."），即匿名投稿形式；Zotero 条目元数据记为 Ning 等 2025。[^src-tsrag]
**Venue**: 缓存仅标注 ICML 匿名投稿，未标注录用信息。[^src-tsrag]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/C8ZF6FKX/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

论文自述的出发点：对 LLM 做 fine-tuning 能适配特定领域但跨未见数据集泛化差且计算开销大；现有 TSFM 缺少内在的领域自适应机制、可解释性有限，不适合 zero-shot 预测[^src-tsrag]。TS-RAG 把 RAG 思路引入 TSFM：用预训练时序编码器从外部知识库检索语义相关的时序片段，以可学习 MoE 增强模块融合检索结果与查询序列，不做任务级 fine-tuning 而提升 zero-shot 精度[^src-tsrag]。

## 方法

框架由 TSFM 原有三件套（encoder、transformer backbone、projection layer）加两个新组件构成：retriever 与 augmentation module[^src-tsrag]。backbone 实验中固定为 Chronos-Bolt，检索编码器用 Chronos encoder，检索库用 FAISS 索引[^src-tsrag]。

- **检索**：知识库存三元组 $(x_i, e_i, y_i)$（context 窗口、其嵌入、对应 future horizon）；对查询嵌入 $e_q$ 按 L2 距离 $d(e_q, e_i)=\|e_q-e_i\|_2$ 取距离最小的 top-k 候选对 $C=\mathrm{TopK}(\cdot, k)$[^src-tsrag]。
- **MoE augmentation module**：作用在检索增强融合处，不替换 backbone 的任何 FFN——TSFM 全部冻结。每个检索到的 future horizon $y_i$ 经可学习 MLP 独立编码 $\hat{e}_i=f_{\mathrm{MLP}}(y_i)$，堆成 $E_{\mathrm{enc}}\in\mathbb{R}^{k\times d}$；与 TSFM 生成的查询表示 $\hat{e}_q$ 拼接为 $E_{\mathrm{concat}}=[\hat{e}_q; E_{\mathrm{enc}}]\in\mathbb{R}^{(k+1)\times d}$。expert 数量为 $k+1$：$k$ 个检索 horizon 的投影嵌入各算一个 expert，加上 1 个查询嵌入自身，即「每个嵌入被当作一个 expert」[^src-tsrag]。拼接表示先过 Multi-Head Attention 得 $E_{\mathrm{att}}=\mathrm{MHA}(E_{\mathrm{concat}})$，再用 gating 网络给每个 expert 打分：$\alpha=\mathrm{Softmax}(W_g E_{\mathrm{concat}} + b_g)$，$\alpha\in\mathbb{R}^{(k+1)\times 1}$。gating 是在全部 $k+1$ 个 expert 上的稠密 softmax，没有 top-k 路由稀疏化（论文中的 $k$ 只是检索条数）；融合时加 skip connection 保留 TSFM 原始输出：$e_{\mathrm{final}}=\hat{e}_q+\sum_{i=1}^{k+1}\alpha_i E_{\mathrm{att},i}$，最后 $\hat{y}_q=f_{\mathrm{proj}}(e_{\mathrm{final}})$ 过 TSFM 输出投影层[^src-tsrag]。
- **训练与超参**：只训练 MoE 增强模块的外部参数；数据为 Chronos 预训练集采样的 5000 万数据点，其中 500 万建检索库，得 2600 万预训练对与 280 万检索对。默认 $k=10$；AdamW、lr 0.0003、weight decay 0.01、batch size 256、10000 steps、dropout 0.2、A6000-48G、TF32[^src-tsrag]。论文没有负载均衡损失，也没有任何辅助损失——全文未出现 load balancing / auxiliary loss 表述[^src-tsrag]。

## 实验

主结果（Table 1，zero-shot，context 512、forecast 长度 64，MSE/MAE，对照 Chronos-Bolt、MOMENT、TTM、Moirai、TimesFM、Chronos）：作者报告 TS-RAG 在全部 7 个数据集上 MSE 与 MAE 均最低；相对 backbone Chronos-Bolt 平均降 3.54% MSE、1.43% MAE；ETTm1 上 Chronos-Bolt 本已较强（MSE 0.3109），TS-RAG 再降 6.51% 至 0.2906。逐数据集 MSE/MAE：ETTh1 0.3557/0.3624，ETTh2 0.2451/0.2982，ETTm2 0.1466/0.2231，Weather 0.1454/0.1771，Electricity 0.1120/0.2002，Exchange rate 0.0627/0.1718[^src-tsrag]。

- **$k$ 敏感性**（Figure 2，$k\in\{1,4,8,16,20\}$，7 数据集 MSE 曲线）：MSE 随 $k$ 增大先显著下降后趋平，个别数据集略回升；Weather/Electricity/Exchange 在 $k>10$ 后收益边际化。论文给出「最优区间 $k$ 在 6 到 10 之间」的取舍结论，并说明更大 $k$ 推高计算成本[^src-tsrag]。
- **检索库版本**（Table 2，MSE）：historical 库（用各数据集自身训练集构建，仅推理期使用）在 7 个数据集中 6 个最低；例外是 ETTh2，pretrained 库 0.2432 优于 historical 0.2451（论文解释多域库在部分场景可能提供更丰富的模式）[^src-tsrag]。
- **检索 lookback**（Table 3，MSE）：64/128/256/512 vs w/o RAG。整体 256 或 512 较好，但论文自述并非越长越好；ETTm1 在 lookback 256 时为 0.3195，差于 w/o RAG 的 0.3109，而 512 时 0.2906 最好——同表内部非单调，此处如实并列[^src-tsrag]。
- **更长 horizon**（Table 4，MSE，96/192/336/720，rolling 策略且每步检索 next 64-step horizon）：作者报告 ETTh1/ETTh2/Weather/Exchange 上 TS-RAG 在全部 4 个 horizon 均低于 w/o RAG，例如 Exchange 720 步 0.8100 对 0.6701[^src-tsrag]。

## 局限

论文没有独立 Limitations 章节。论文自述的限制散见于正文：检索 lookback 过长可能引入噪声或不相关信息，需要自适应检索机制；$k$ 增大有计算成本；Table 1 中 Weather 与 Electricity 对部分 TSFM 标 "—"（数据被用于预训练、不报告 zero-shot 结果），对照面不完整[^src-tsrag]。结论段给出的未来方向（多模态扩展、检索排序优化）为论文自述的改进空间，非实证结论[^src-tsrag]。

## 与其它 MoE 方法的差异

论文将 MoE 出处引至 Shazeer et al. 2017（sparsely-gated mixture-of-experts layer），并把本文的用法定位为「可学习 MoE 增强模块」：expert 不是 backbone 内的 FFN 分支，而是嵌入级的融合通道（每个检索 horizon 投影嵌入 + 查询嵌入），gating 为稠密 softmax，与原文的稀疏 top-k gating 不同，也没有负载均衡机制。论文没有与其它 MoE 时序模型（如 Moirai-MoE 类方法）做实验对比，未提及此类比较[^src-tsrag]。

[^src-tsrag]: [[source-tsrag]]
