# Ingest 报告：Deep Learning for Multivariate Time Series Imputation: A Survey

**日期**: 2026-08-29
**源文件**: `raw/wang-mts-imputation-survey-arxiv-2024.pdf`（arXiv:2402.04059v3 [cs.LG]，2025-05-20，9 页）
**著录核实**: PDF 水印为 arXiv v3（20 May 2025），PDF 内无 IJCAI 2025 接收标识（"IJCAI" 仅出现在参考文献与 Table 1 的他人论文 venue 列）。IJCAI 2025 著录来自用户，未在 PDF 内核实——已在 source-summary 页如实记录。

## 创建

- `wiki/source-mts-imputation-survey.md` — WHY：每个 raw 文件必须有对应 source-summary；记录综述的覆盖范围、双视角分类框架、三大贡献（分类法/工具箱/未来方向）与作者自述边界，以及 PDF 版本核实结论。
- `wiki/mts-imputation-taxonomy.md` — WHY：综述的核心可复用贡献是其"插补不确定性（预测式/生成式）× 网络架构（含大模型单列）"分类框架；将其沉淀为概念页后，可为后续所有插补类 ingest 提供"综述口径"的归类锚点，并集中记录本 wiki 各方法在该框架中的位置（含二手归因警示）。
- `wiki/pypots.md` — WHY：PyPOTS 生态与 TSI-Bench 是综述三大自述贡献之一，且 wiki 此前无任何页面提及 PyPOTS（实体缺口）；后续插补论文 ingest 时可复用该链接目标。

## 修改

以下页面各追加一节"综述归类"（或定位条目），统一使用「综述认为/综述 Table 1 标注」的归因口径，不替换原论文口径的数字与细节：

- `wiki/csdi.md` — WHY：综述归类 CSDI 为生成式-扩散类并称其为首个专门为 MTSI 设计的扩散模型，其双 Transformer 二次复杂度概括与本页局限章节互证；source_count 12→13。
- `wiki/pristi.md` — WHY：综述归类 PriSTI 为生成式-扩散类（Diffusion+Attention+GNN+CNN），条件机制概括与本页"先验引导注意力"口径一致；source_count 6→7。
- `wiki/grin.md` — WHY：综述称 GRIN 为首个基于图的循环 MTSI 架构（预测式-GNN 类），加强本页"首个 GNN 填补模型"论断；source_count 4→5。
- `wiki/imputeformer.md` — WHY：综述将其归为预测式-Attention 类，但转述未涉及其低秩核心——显式标注两套口径分立，防止后续读者误用二手描述；source_count 6→7。
- `wiki/sadi.md` — WHY：综述概述其为相似度感知扩散模型，且 Table 1 是 33 个方法中仅有的两个 MNAR 相关标注之一（MCAR/MAR/MNAR）；source_count 1→2（同时缓解该页 confidence: high 与 source_count: 1 的 lint 隐患）。
- `wiki/nuwats.md` — WHY：综述将 NuwaTS 放入大模型（PFM）类，与原论文"首个跨域插补基础模型"自述互证；source_count 4→5。
- `wiki/timesnet.md` — WHY：综述从插补任务视角将 TimesNet 单独归为预测式-CNN 类，与原论文"五任务通用架构"口径分立，需显式区分；source_count 3→4。
- `wiki/missing-not-at-random.md` — WHY：综述给出与 PRDIM 口径一致的 Rubin 框架，并补充领域层面论断（现有方法多以 MCAR/MAR 运作、MNAR 列为未来方向之首、Table 1 中仅 supnotMIWAE/SADI 涉及 MNAR）；source_count 2→3。
- `wiki/two-stage-imputation.md` — WHY：命名冲突预防——综述的 "impute and predict" 两阶段范式（流水线级：插补+下游模型）与本页"双阶段插补"（模型内精炼）不同义，新增命名辨析章节；source_count 2→3。
- `wiki/fence.md` — WHY：FENCE 是引用本综述的后续论文之一（AAAI 2026）；按综述框架给出分析性定位（生成式-扩散类），并标注"综述未收录 FENCE、此为 wiki 分析"的边界；source_count 4→5（与并行代理的 lets-group/costi 追加合并）。
- `wiki/loft.md` — WHY：LOFT 是引用本综述的后续论文之一（KDD 2026）；按综述框架给出分析性定位，并标注综述未覆盖流匹配路线；source_count 1→2。

## 新建交叉链接

- [[source-mts-imputation-survey]] ↔ [[mts-imputation-taxonomy]]、[[pypots]]
- [[mts-imputation-taxonomy]] ↔ [[csdi]]、[[pristi]]、[[grin]]、[[imputeformer]]、[[sadi]]、[[nuwats]]、[[timesnet]]、[[loft]]、[[giflow]]、[[fence]]、[[missing-not-at-random]]、[[two-stage-imputation]]、[[pypots]]
- [[missing-not-at-random]] ↔ [[mts-imputation-taxonomy]]
- [[two-stage-imputation]] ↔ [[mts-imputation-taxonomy]]
- [[pypots]] ↔ [[csdi]]、[[grin]]、[[imputeformer]]、[[tslib]]

## 矛盾处理

未发现需要 `## 争议` 或 `## 历史论断` 的实质矛盾。逐页核对结果：

1. GRIN："首个图神经网络填补模型"（原文口径）vs 综述"首个基于图的循环 MTSI 架构"——措辞差异（综述多出 "recurrent"），互相印证，无冲突。
2. CSDI："首个条件扩散插补"（原文口径）vs 综述"首个专门为 MTSI 设计的扩散模型"——一致。
3. SADI：综述 Table 1 标注其机制为 MCAR/MAR/MNAR，原论文口径是 partial blackout 模式——非冲突但为二手归类，已用"两套口径分立"处理并保持归因。
4. ImputeFormer：综述转述缺失其低秩核心——按「不用综述二手描述替换原文细节」原则，仅记录归类并显式标注转述局限。
5. TimesNet：综述单任务归类（预测式插补/CNN）vs 原文五任务通用架构——记录为口径分立。
6. two-stage-imputation：命名撞车（模型内双阶段 vs 综述的 impute-and-predict 流水线）——以命名辨析章节处理，未合并概念。

## 未建页面说明（工作流第 4 步判断）

- SAITS：综述正文明述其双任务自监督设计，且 [[sadi]]/[[two-stage-imputation]] 等页已链接 [[saits]]，但 wiki 尚无 saits.md——超出本综述 ingest 范围（综述引用的每篇论文不逐一建页），留待 SAITS 原文 ingest。
- 综述提及的 GRU-D、BRITS、SPIN、GP-VAE、SSSD、CSBI、MIDM、FGTI、MTSCI、SPD、MOMENT、Timer、GPT4TS 等均未建页，原因同上；其综述归类已集中记录于 [[mts-imputation-taxonomy]]。
- TimeGrad/COSTI 相关：综述未提及 TimeGrad；costi 页面属并行 ingest 代理的新建文件，本代理未触碰。

## Frontmatter 更新

所有修改页面的 `last_updated` 均为 2026-08-29；`source_count` 按实际新增 `[^src-mts-imputation-survey]` 引用递增；confidence 未上调（各页主要论断仍以原论文为准，综述为单一补充来源）。

## 审查补记（2026-08-29 lint-fix）

对照 `raw/wang-mts-imputation-survey-arxiv-2024.pdf` 全文（pdftotext，Table 1 另以逐行对位核对）复核本 ingest，以下为审查发现与修复：

1. **贡献编号错误（已修）**：`source-mts-imputation-survey.md` 原写"综述第三项贡献是工具箱梳理"，PDF 贡献列表中工具箱为第 2 项（1 分类法 / 2 工具箱 / 3 未来方向），已改为"第二项贡献"。
2. **术语措辞（已修）**：source-summary 原写 "impute-then-predict"，综述原文为 "impute and predict"（Sec. 7），已改；`two-stage-imputation.md` 本就标注了原文措辞，无误。
3. **引用溯源（已修）**：`fence.md`/`loft.md` 原以"据本仓库 ingest 著录"声称"FENCE/LOFT 论文引用了该综述"且挂 `[^src-mts-imputation-survey]`（引用错配——综述 PDF 无法佐证他人论文引用它）。复核结果：FENCE 引用属实（`raw/fence-...aaai26.pdf` 参考文献含 "Wang, J.; Du, W.; ... 2024. Deep learning for multivariate time series imputation: A survey. arXiv:2402.04059"），`fence.md` 改挂 `[^src-fence]` 并注明 raw PDF 核实；LOFT 在仓库内不可核实（raw/ 无 LOFT PDF、`source-loft` 未记录其参考文献），`loft.md` 与 `mts-imputation-taxonomy.md` 已改为"未在仓库内核实"。
4. **交叉链接补全（已修）**：本报告"新建交叉链接"声称与 [[giflow]] 双向互链，实际 giflow.md 未被本 ingest 修改；已在 `giflow.md` 相关页面补 [[mts-imputation-taxonomy]] 指向（纯结构链接，未动并行代理对 giflow 的改动）。
5. **字数（已修）**：source-summary 压缩至 300-500 汉字区间（复核 528→≤500，与 RDPI lint 口径一致）。
6. **source_count 合并终值备忘**：报告中"X→Y"记录本 ingest 自身增量；因并行 CoSTI/Let's Group ingest 同日追加引用，最终合并计数为 csdi 14、grin 6、pristi 8、loft 3（本 ingest 修复后又 +0，[^src-costi] 来自并行）、fence 5、giflow 2、imputeformer 7、sadi 2、nuwats 5、timesnet 4、missing-not-at-random 3、two-stage-imputation 3；`mts-imputation-taxonomy` 因补 [^src-fence] 由 1→2。其余核对结论：Table 1 确为 33 个方法（12 预测式/16 生成式/5 大模型），MNAR 标注仅 supnotMIWAE（MNAR）与 SADI（MCAR/MAR/MNAR）；PyPOTS 37 模型、TSI-Bench 172 数据集/28 算法/34,804 实验均出自综述 Sec. 6 原文，引用挂接无误；GRIN、CSDI、PriSTI、ImputeFormer、SADI、NuwaTS、TimesNet、two-stage 各页"综述归类"转述逐条与 PDF 一致且归因口径齐全。
