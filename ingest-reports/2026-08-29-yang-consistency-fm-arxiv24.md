# Ingest 报告：Consistency-FM — Defining Straight Flows with Velocity Consistency（arXiv:2407.02398）

日期：2026-08-29
源文件：`raw/yang-consistency-fm-arxiv24.pdf`（arXiv v1，2024-07-02；本次会话从 arxiv.org/pdf/2407.02398 下载入库——LOFT ingest 时 raw/ 未包含此引用原文）
背景：LOFT/FENCE 引用分析（query 会话）将本文列为 LOFT 特有未收录引用首位；用户指示"按 AGENTS.md 审查并修复"，故先完成论文口径审查（全文核对 PDF，含 Eq. 6、Lemma 1、Theorem 1/2、Table 2/3 数字、preliminary 措辞与 [41]=SD3 引文），再执行 Ingest 工作流。

## 创建

- wiki/source-yang-consistency-fm-arxiv24.md — WHY：source-summary 页，记录问题/机制（Lemma 1 等价、双项损失 + EMA 目标、多段线性化、蒸馏变体）/证据（CIFAR-10、AFHQ-Cat 无条件生成 FID，作者自述 preliminary）/范围（无时序插补实验、预印本接收状态未核实）
- wiki/consistency-fm.md — WHY：实体页，汇集方法内容，并显式记录与 LOFT $L_{CT}$ 的四点实现差异（EMA vs stop-gradient、f 项有无、α 符号冲突、多段线性化未采用），防止后续 agent 将两者混同

## 修改

- wiki/trajectory-consistency-flow-matching.md — WHY：对照表行补 [[consistency-fm]] 链接；新增"与 Consistency-FM 原始损失的差异"小节（含 Lemma 1 ↔ LOFT Lemma 4.1 同一等价关系注记）；source_count 1→2，last_updated 更新
- wiki/consistency-models.md — WHY：LOFT 段落内 Consistency-FM 首次出现处补链接（消除"提及但无页面"）；相关页面补入口
- wiki/uncertainty-aware-rectification.md — WHY："与静态矫正的关系"首句补链接；相关页面补入口
- wiki/loft.md — WHY：相关页面补 [[consistency-fm]]（$L_{CT}$ 方法来源）；frontmatter 由并发会话已更新，未重复改动
- wiki/index.md — WHY：文件尾部新增 1 个 source、1 个 entity 条目（避开并发会话的编辑区域）
- wiki/log.md — WHY：记录本次 ingest 与审查结论

## 新建交叉链接

- [[consistency-fm]] ↔ [[trajectory-consistency-flow-matching]]
- [[consistency-fm]] ↔ [[consistency-models]]
- [[consistency-fm]] ↔ [[uncertainty-aware-rectification]]
- [[consistency-fm]] ↔ [[loft]]
- [[consistency-fm]] ↔ [[rectified-flow]] / [[shortcut-models]] / [[flow-matching]]（相关页面单向入口）

## 口径与边界说明

- 论文内容的所有论断归因为论文自述；实验数字限定在"无条件图像生成、作者报告、自述 preliminary"范围内。
- venue：arXiv v1（2024-07-02）后未更新；OpenReview 提交记录（bS76qaGbel）存在但接收状态未核实，页面按"arXiv 2024 预印本"著录，置信度 source 页 low / entity 页 medium（2 源无反驳，第二源仅覆盖与 LOFT 的关系）。
- 未为 MeanFlow / AlphaFlow / MSFM / FGTI 建页：同为 LOFT 少步生成来源或基线，待源文件入库后另行 ingest。
- 并发说明：本会话与另一 ingest 会话（FBG、GiFlow）并行工作；index/log 编辑均采用文件尾部追加锚点，loft.md frontmatter 检测到并发更新后未重复改动。
