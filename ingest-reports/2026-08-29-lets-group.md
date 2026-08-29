# Ingest 报告：Let's Group — A Plug-and-Play SubGraph Learning Method for Memory-Efficient Spatio-Temporal Graph Modeling（raw/weng-lets-group-ijcai-2025.pdf）

日期：2026-08-29。执行：ingest 工作流第 1–9 步（并行安全规则下，index/log 最后追加）。

## 版式与著录核实

- PDF 为 **IJCAI-25 官方 proceedings 排版**：每页页眉 "Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)"，页码 3471–3479；`pdfinfo` Subject 字段为 "Paper accepted and presented at IJCAI-2025"，共 9 页。
- 论文全题（以 PDF 为准）：**Let's Group: A Plug-and-Play SubGraph Learning Method for Memory-Efficient Spatio-Temporal Graph Modeling**（比用户著录的短题名长，含 "Memory-Efficient Spatio-Temporal Graph Modeling" 副题）。
- 用户著录的论文集编号 **No. 386 未在 PDF 正文与元数据中出现**（全文与 pdfinfo 均无该编号）；会议著录本身（IJCAI-25、页码区间）已核实，另经 FENCE（AAAI-26）参考文献条目 "IJCAI-25, 3471–3479" 交叉印证。
- 核心机制以 PDF 为准确认：SGL 用于**时空图建模/交通预测**的内存开销降低（SGPM 子图划分 + SGFAM 跨子图特征聚合）；"即插即用" 指模块可直接接入现有 STGNN 的空间特征提取环节（Sec. 3.3/3.5）。论文与插补无关。

## 创建

- `wiki/lets-group.md` — WHY：论文的核心方法页（type: technique）。记录问题（O(N²) 相关矩阵内存瓶颈）、机制（SGPM 记忆向量锚点 top-K 划分、子图共享 G()、SGFAM 平均聚合）、复杂度 O(N+K²)、实验（4 个 PEMS 数据集 × 8 个 backbone，Table 1–3）、消融（Table 4–5）、超参（Fig. 4）、可视化（Fig. 5），以及适用范围与被引口径差异。所有数字与论断挂章节/表号，作者报告口径。未建子页面：SGPM/SGFAM 是同一流水线的两个模块，单页足以承载，拆分会碎片化一篇 7 页正文论文。
- `wiki/source-lets-group.md` — WHY：raw 文件对应的 source-summary（正文 307 汉字，300–500 区间内），注明 raw 文件名、版式核实结论与 No. 386 未在 PDF 内出现的口径。

## 修改

- `wiki/patchstg.md` — WHY：本文相关工作将 PatchSTG 的地理坐标分块归为静态划分（Sec. 1，其参考文献引 PatchSTG 的 arXiv:2412.09972 版本）；在 Context 节补记该定位与 SGPM 的对照方案。source_count 2→3，last_updated 2026-08-29。
- `wiki/memory-efficient-training.md` — WHY：SGL 是建模层面的内存优化（相关矩阵 N×N→K×K），与本页系统级技术（激活检查点、DiffusionBlocks、混合精度）作用层面不同，新增小节与 Related Concepts 条目。source_count 1→2，last_updated 2026-08-29。
- `wiki/large-scale-spatial-temporal-graph.md` — WHY：「结构感知方法 → 图划分」类目下补 SGL 条目（可学习划分 vs 静态划分），相关页面清单补链。source_count 5→6，last_updated 2026-08-29。
- `wiki/traffic-forecasting.md` — WHY：任务总览页 "Spatial Patching / Efficient Dynamic Spatial Modeling" 节后新增 "Learnable Subgraph Partitioning / Memory-Efficient STGNN" 小节，衔接大规模效率方法脉络。source_count 50→51，last_updated 2026-08-29。
- `wiki/node-visibility.md` — WHY：VisiFold 的随机子图采样与 SGL 的可学习划分构成同类机制的两条路线（随机/正则化 vs 相似度/降内存+聚合），Related Pages 补对照条目。source_count 1→2，last_updated 2026-08-29。
- `wiki/staeformer.md` — WHY：STAEformer 是本文 8 个 backbone 之一（STAEformer-SGL：PEMS07 GPU cost 22.11→9.34 GB、PEMS07 MAE 19.22→19.16，Table 3），补记该实验角色。source_count 2→3，last_updated 2026-08-29。
- `wiki/fence.md` — WHY：FENCE 是本文的被引来源。fence.md 原未提及 Let's Group；在「另见」补条目并记录引用语境差异（见下）。source_count 3→4，last_updated 维持 2026-08-29。
- `wiki/source-fence.md` — WHY：FENCE Related Work 将 "(Cao et al. 2018; Che et al. 2018; Weng et al. 2025)" 并列归入判别式插补模型，Weng et al. 2025 即本文；新增「相关工作引用口径」小节如实记录该归类与本文原文（交通预测、无插补实验）的差异。source_count 1→2，last_updated 2026-08-29。

## 被引口径差异（非矛盾，未触发矛盾解决策略）

FENCE（AAAI-26）将本文列入「判别式时空插补模型」，但本文的任务设定（Sec. 3.1）与全部实验（Sec. 4, Table 3）均为交通预测，无插补实验。处理：两侧分别归因记录于 [[source-lets-group]]、[[source-fence]]、[[fence]]、[[lets-group]]，以本文原文口径为准；未设 `status: disputed`（引用语境归类差异，非对同一事实的冲突论断）。

## 矛盾检查

- [[patchstg]] 的「First to bridge KDTree and Transformer patching」为 PatchSTG 自述且范围限于 KDTree+Transformer patching，与本文不冲突。
- [[traffic-forecasting]] 的三分类（Linear/Low-rank/Patching）出自 PatchSTG 框架，SGL 以独立小节并列，不改动原分类口径。
- 无其他页面的既有论断与本文冲突。

## 新建交叉链接

- [[lets-group]] ↔ [[source-lets-group]]
- [[lets-group]] ↔ [[patchstg]]（静态划分 vs 可学习划分）
- [[lets-group]] ↔ [[memory-efficient-training]]（建模层 vs 系统层内存优化）
- [[lets-group]] ↔ [[large-scale-spatial-temporal-graph]]（图划分/大规模方法类目）
- [[lets-group]] ↔ [[traffic-forecasting]]（任务总览）
- [[lets-group]] ↔ [[node-visibility]]（随机 vs 可学习子图划分）
- [[lets-group]] ↔ [[staeformer]]（实验 backbone）
- [[lets-group]] ↔ [[fence]] / [[source-fence]]（被引来源与口径差异）
- [[lets-group]] ↔ [[testam]]、[[std-mae]]（本文相关工作的提及，单向）

## 遗留

- LarSTL（Wang et al., IJCAI-24, "Make bricks with a little straw"）与本文同属受限 GPU 内存下的时空图建模，raw/ 无该文 PDF，未建页；已在 [[lets-group]] 相关工作语境中提及。
- 本文正文将 Lee and Ko 2024 的 MoE 工作称为 "EXPERT"（参考文献条目为 TESTAM）；[[lets-group]] 按论文正文记法标注并括注对应 wiki 页面 [[testam]]。

## 审查补记（2026-08-29，审查与修复一体）

审查代理对照 PDF 全文（pdftotext 与 -layout 双版本）逐条复核：Table 1–5 全部数字（数据集节点数/时间步、M/K 设置、Table 3 的 15 行 MAE/RMSE/MAPE/GPU cost/train time、Table 4–5 消融、PEMS07 GPU 22.11→9.34 GB、DDGCRN 2.85→2.33/12.59→4.97 与 18.2%/60.5%、GMAN 16.22→4.71 与 166.77→80.48、DGCRN 106.95→125.32）、Eq. 5–12、复杂度推导 O(N²)→O(NM+MK+K²+N)→O(N+K²)、Fig. 4–5 描述、EXPERT（Lee & Ko 2024）参考文献实为 TESTAM 的注记，均与论文一致，未发现编造数字。FENCE 引用口径核实：FENCE 原文 Related Work（Spatial-Temporal Imputation 段）作 "Discriminative models (Cao et al. 2018; Che et al. 2018; Weng et al. 2025)"，参考文献 Cao 2018 = BRITS、Che 2018 = GRU-D 论文（Scientific Reports 8(1):6085）、Weng et al. 2025 著录 IJCAI-25, 3471–3479 与本 PDF 一致；四处（[[source-lets-group]]、[[source-fence]]、[[fence]]、[[lets-group]]）均以本文原文口径为准、归类写作 FENCE 的语境。修复 6 处：

1.（严重）[[lets-group]] 被引口径段 "GRIN-D" 为错误模型名——FENCE 引的 Che et al. 2018 是 GRU-D（"GRIN-D" 系与 GRIN 混淆），改为 BRITS（Cao et al. 2018）、GRU-D（Che et al. 2018）并标注对应关系。log 的 ingest 条目按 append-only 不回改，以本条为准。
2.（警告）[[lets-group]] source_count 1 与正文实际唯一源数不符（正文用了 [^src-lets-group]、[^src-visifold]、[^src-fence]）→ 改 3。
3.（警告）[[source-lets-group]] source_count 1 与实际（[^src-lets-group]、[^src-fence]）不符 → 改 2。
4.（警告）[[lets-group]]「PatchSTG 用 leaf KDTree 按地理坐标做不规则空间分块（KDD 2025）」出自 PatchSTG 侧来源、原文只说 geographical coordinates——补 [^src-patchstg] 分源引用，source_count 3→4。
5.（警告）[[lets-group]]「与 [[memory-efficient-training]] 作用在不同层面」为跨论文课程对照，原文无此表述——移出 [^src-lets-group] 引用范围并标注「本课程层面的对照，非论文原文表述」。
6.（信息）[[lets-group]] Table 5 段补 PEMS08 数据与唯一的反向格（sum MAPE 9.00% vs 平均 9.01%），注明论文正文「平均结果最佳」未讨论该例外；另将「相关矩阵构造与特征加权的复杂度均为 O(N²)」两处措辞改为论文原口径（O(N²) 属于图卷积/注意力机制，复杂度因两步而与 N 耦合），「划分子网」改「划分子图」（原文 subgraphs）。
7.（严重，二次审查更正）Table 3 数字复核的更正：第 6 条前次结论「Table 1–5 全部数字均与论文一致」对 STAEformer 一处不成立——Table 3 中 PEMS07 的 STAEformer MAE 实为 19.22→19.16（-layout 版 STAEformer 行 19.22/32.72/8.04%/22.11），18.25→18.29 是 PEMS04 的数字；原稿把它与 PEMS07 GPU cost 22.11→9.34 GB 并列且未标数据集（[[staeformer]] 原文明确写作 "PEMS07 MAE 18.25→18.29"，属数据集错标而非编造），[[lets-group]]、[[staeformer]]、本报告修改节三处已同步更正。另做两处信息级修正：[[source-lets-group]]「性能追平原模型」改「性能与原模型相当」（论文原文 "matches that of DDGCRN"，避免体育化表达）；同页局限节「串行 RNN 型 backbone 无并行加速收益」改为论文原口径「不产生并行加速，其 SGL 变体运行效率与原型相当、大节点规模时更优」（Sec. 4.1）。

另核：[[source-lets-group]] 正文 307 汉字（去作者/发表/代码元数据行后的正文口径）与 389 汉字（含元数据行）均在 300–500 区间；摘要「平均 GPU 内存开销最高降 56.4%」论文未给出平均口径，页面已如实标注、未代论文补算。未触碰 CoSTI/MTS 综述相关文件与条目。
