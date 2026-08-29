# Ingest 报告：DiffPuter (Zhang et al., ICLR 2025)

**源文件:** `raw/zhang-diffputer-iclr-2025.pdf`
**版式核实:** PDF 每页页眉为 "Published as a conference paper at ICLR 2025"，首页带 arXiv:2405.20690v2 (24 May 2025) 水印——确认为 ICLR 2025 camera-ready 版，会议著录已在 PDF 内核实，无需依赖用户著录。
**日期:** 2026-08-29

## 创建

- `wiki/diffputer.md` — WHY：核心技术页。DiffPuter 是 EM+扩散插补框架（M 步密度估计 / E 步条件采样），PRDIM 页面已三次引用其名但无目标页，且 LOFT 所在的插补方法族需要这个扩散+EM 路线的锚点页。
- `wiki/source-diffputer.md` — WHY：每份 raw 源文件必须有对应 source-summary；记录 PDF 版式核实结果、论文自述定位、实验数字（含数据集/基线计数不一致的如实记录）与局限。
- `wiki/em-diffusion-interleaving.md` — WHY：论文的独特机制概念（EM 两个步骤分别落到扩散模型的训练与采样，M 步=MLE、E 步=EAP），值得独立技术页以便 PRDIM/后续 EM 插补工作引用。

## 修改

- `wiki/csdi.md` — WHY：在"后续影响"补充表格数据方向的谱系（TabCSDI 为 CSDI 表格适配、DiffPuter 以其为扩散基线并报告明显占优），补交叉链接与 `[^src-diffputer]`；source_count 10→11。
- `wiki/prdim.md` — WHY：PRDIM 是 EM+扩散路线的后续；将"vs DiffPuter"一句升级为带 DiffPuter 论文自述定位（首个扩散+EM，`[^src-diffputer]` 归因）的对照，补交叉链接；source_count 1→2。
- `wiki/pattern-recognizer-guidance.md` — WHY：正文"DiffPuter 的 soft EM"处补 wikilink，让读者可核对 DiffPuter 原文；仅结构性链接，不新增事实引用。
- `wiki/missing-not-at-random.md` — WHY：DiffPuter 沿 Muzellec et al. 2020 协议在 MCAR/MAR/MNAR 三机制下评测但不建模缺失过程，是"各模型缺失机制假设"表的自然新增行；source_count 1→2。

## 未修改及理由

- `wiki/loft.md` — 任务前提称 LOFT 把 DiffPuter 作为扩散插补基线，但：(1) loft.md 及 source-loft.md 记录的 11 个基线（IGNNK/GCASTN/ImputeFormer/LCR/CSDI/PriSTI/FGTI/MTSCI/CoFill/MSFM/FENCE）中无 DiffPuter；(2) source-loft.md 标注的 raw 文件 `raw/loft-low-rank-prior-induced-consistency-flow-matching-efficient-traffic-imputation.pdf` 在 raw/ 目录中不存在，无法核实 LOFT 正文是否在相关工作处引用 DiffPuter。按"论文来源受限时保留可核实内容、禁止猜测"原则，不改 loft.md。
- `wiki/grin.md`、`wiki/pristi.md`、`wiki/cofill.md`、`wiki/ssd-ts.md` — DiffPuter 是表格数据插补，未与这些时空插补方法直接对比（其图基线 IGRM（Zhong et al., AAAI 2023，迭代图重建、表格数据）与 GRIN（ICLR 2022）是不同工作，已避免混淆）；交叉统一由 csdi.md 枢纽承担。
- 无"首个/最优"冲突：CSDI 页的"首个"限定于条件扩散+多元时间序列插补，DiffPuter 的"首个"自述限定于扩散+EM 组合，范围不同，未触发矛盾解决策略。

## 新建交叉链接

- [[diffputer]] ↔ [[source-diffputer]]
- [[diffputer]] ↔ [[em-diffusion-interleaving]]
- [[diffputer]] ↔ [[csdi]]（表格方向谱系）
- [[diffputer]] ↔ [[prdim]]（soft EM vs hard EM + 模式识别器）
- [[diffputer]] ↔ [[pattern-recognizer-guidance]]（hard/soft EM 对照）
- [[diffputer]] ↔ [[missing-not-at-random]]（缺失机制评测协议）

## 记录

log.md 已追加 ingest 条目；index.md 已追加 `[[source-diffputer]]`（Sources）、`[[diffputer]]`（Entities）、`[[em-diffusion-interleaving]]`（Techniques）条目。

**收尾运行补记（2026-08-29）**：前次运行在写入 index.md/log.md 前中断，两个共享文件的条目由收尾运行补齐（追加前重新 Read 尾部确认无冲突）。收尾运行另做一次全文 PDF 复核：diffputer.md / source-diffputer.md / em-diffusion-interleaving.md 中的全部实验数字（图 2 题注 6.94%/4.78%、表 1 Accuracy 62.82 与 Remasker 62.06、表 2 训练时间与 8%-25%、表 3、表 6 题注 13.37%/4.43% vs 表内 13.09%/4.60%、图 3-6 消融、表 9 MNAR Adult ReMasker 47.66 vs DiffPuter 48.59）与 raw/zhang-diffputer-iclr-2025.pdf 一致；顺手将 em-diffusion-interleaving.md 数学环境内 `\*` 转义改为 `^*`（对齐全 wiki 惯例）。注：csdi.md 的 source_count 现为 12（本次 ingest 贡献 10→11，随后并行 ingest（RDPI）的引用再 +1，两处引用均经核实存在）。

## 审查修复补记（2026-08-29，lint-fix）

独立审查（对照 PDF 全文逐数字核对）后修复：

- `wiki/diffputer.md` — type entity→technique（内容为方法机制页，与 csdi.md 同类；AGENTS.md 中 entity 限人物/组织/产品/地点）；index.md 条目随之由「实体」移入「技术」。表 3 行改为分立归因：论文正文概括 "leads to performance improvements" 与表 3 数据并存记录（EM+HIWAE 在 Default/Shoppers/News 三列劣于 HIWAE 单模型，如 Default 0.4314 vs 0.3989）。图 6/局限统一记录论文两种措辞（正文 "performance lower bound" vs 题注 "upper-bounded by mean imputation"），消除原页面一处「下界」一处「上界」且未注明出处的自相矛盾。表 6 行 IGRM「全部数据集失败」标注为第 5.2 节正文口径并补表内实况（5 列给出大幅劣化数值、Default 列 OOM）。补记论文内部不一致两处：表 1 题注 "five datasets" 实列 4 个；附录 D.2 写 "ten" 且称五个混合特征数据集却列 4 个。TabCSDI 谱系（CSDI 表格适配）改标为 wiki 依名称所作推断，非论文原文表述。默认 N=10 的出处改标第 5.3 节与附录 D.4。
- `wiki/source-diffputer.md` — 脚注定义改为 `[^src-diffputer]: [[source-diffputer]]`（对齐 AGENTS.md `[^src-X] → [[source-X]]` 惯例与其他 source 页；raw 路径信息保留在页首）。
- `wiki/prdim.md` — 「vs DiffPuter」句的 `[^src-diffputer]` 引注紧跟「首个」自述，"soft EM" 明确标注为 PRDIM 转述口径。
- `wiki/missing-not-at-random.md` — 协议出处补 Zhao et al. (2023)（MAR/MNAR 掩码生成具体沿用该文，DiffPuter 附录 D.3）。
- `wiki/em-diffusion-interleaving.md` — 审查无需改动（机制公式、Theorem 1/Remark 2 表述与 PDF 一致，`\*` 修复完整）。
