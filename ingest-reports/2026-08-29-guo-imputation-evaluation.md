# Ingest 报告：Guo et al., A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models（arXiv:2412.04733v2）

日期：2026-08-29。源文件：`raw/guo-imputation-evaluation-st-traffic-arxiv-2024.pdf`（只读，未修改 raw/）。评测型 ingest：价值在于为 wiki 现有交通插补方法页面补"评测口径"引用，并建立评测协议/结论的锚点页面。

## 版本核实

- PDF 水印：arXiv:2412.04733**v2** [cs.LG] 17 Oct 2025；题名 "A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models"（v2 改题后题名，PDF 内核实成立）。
- v1 题名依 FENCE（AAAI 2026）参考文献著录为 "An Experimental Evaluation of Imputation Models for Spatial-Temporal Traffic Data"（2024；raw/fence-*.pdf 内核实）。本仓库无 v1 PDF。
- 用户著录 "v2 改题" 与 PDF 一致。

## 创建

- wiki/source-guo-imputation-evaluation.md — WHY：每个 raw 文件对应一个 source-summary；记录版本核实、评测范围与主要发现，是 `[^src-guo-imputation-evaluation]` 的引用锚点。
- wiki/st-traffic-imputation-benchmark.md — WHY：评测自身贡献（统一管线协议、11 模型清单、按数据集×缺失模式的排名结论、效率与挑战期分析、模型选择建议）需要一个独立概念页承载，供各方法页面回填评测口径时引用；并明确"评测复现数字与原文数字分立"的口径规则。
- wiki/traffic-missing-patterns.md — WHY：SRTR/SRTC/SCTR/SCTC 四分类是可独立引用的评测贡献（缺失模式的几何结构分类），与 [[missing-not-at-random]] 的 Rubin 机制分类维度不同，分立记录。

## 修改

- wiki/pristi.md — WHY：PriSTI 在评测清单内，页面缺评测口径。新增"统一评测口径"节：低缺失率表现好、最常进 top-3 之一（评测者归因先验引入）、推理 8553.77s 最慢、性能-内存平衡定位；source_count 10→11。
- wiki/imputeformer.md — WHY：ImputeFormer 在评测清单内。新增评测口径节：top-4 深度模型、PEMS04/08 SR- 模式最好（SC- 下 LATC 最好）、挑战/稳定期前三、效率数字与定位；source_count 7→8。
- wiki/std-plm.md — WHY：STD-PLM 在评测清单内，原页面仅 1 源。新增评测口径节：top-4、挑战/稳定期前三、内存 8744MB 最大；source_count 1→2；confidence high→medium（对齐规则：原先 1 源标 high 不合规，现 2 源无反驳取 medium）。

## 判断后不修改的页面

- grin.md / csdi.md — 综述正文提及（STGI/TSPI 类代表）但未进入 11 模型评测清单，无评测结论可回填；仅在评测页注明"提及未评测"。
- fence.md / loft.md / mtsci.md / diffputer.md / costi.md / rdpi.md / lcr.md / fgti.md / giflow.md 等 — 均不在评测清单（评测模型为 2025-10 v2 之前工作），无冲突；与 LOFT/FENCE 各自协议的对比数字已在各自页面按分立口径记录，本 ingest 不重复。
- mts-imputation-taxonomy.md — 该综述归类已由先前 ingest 处理；Guo 评测的模型分类与其分类法不同，经 [[st-traffic-imputation-benchmark]] 间接关联即可，不做条目合并。

## 冲突核对（矛盾解决策略）

- 无同协议直接矛盾，未触发情况 1/2/3，无页面 status 变更。
- 口径差异（非矛盾，已分立）：
  - PriSTI 原文报告对 BRITS/GRIN/CSDI 的优势（其自设协议）vs 该评测复现口径下 BRITS 在 Seattle 一致最优、PriSTI 未在任一数据集整体第一——数据集与协议不同，两套口径分立记录，不替任何一方圆场。
  - ImputeFormer 原文 "10 基准 SOTA"（自设协议）vs 评测复现口径下 PEMS04/08 SR- 模式最好、SC- 模式 LATC 最好——排名限定协议范围，未写成普遍事实。
- 论文自身不一致（如实记录，不替论文选数字）：
  - 摘要/贡献写 11 模型 vs 正文 Sec. IV 与 V.B 三处写 "10"（"select the following 10 models"、"10 recently proposed sequence imputation and prediction models"、"10 baselines"；实际列出 A–K 共 11 个，另有 LAST 基线）。
  - 正文称 LATC 训练时间最小 vs 其 Table IV 中 E2GAN 312.42s < LATC 829.00s。
  - 正文称深度方法 E2GAN 推理最快、BRITS 次快 vs Table IV 中 IGNNK 1.59s < BRITS 17.46s。

## 新建交叉链接

- [[source-guo-imputation-evaluation]] ↔ [[st-traffic-imputation-benchmark]]
- [[st-traffic-imputation-benchmark]] ↔ [[traffic-missing-patterns]]
- [[st-traffic-imputation-benchmark]] ↔ [[pristi]]、[[imputeformer]]、[[std-plm]]（评测口径回填）
- [[traffic-missing-patterns]] ↔ [[missing-not-at-random]]（几何结构分类 vs 统计机制分类，分立）
- [[traffic-missing-patterns]] ↔ [[partial-blackout]]、[[message-passing-imputation]]、[[loft]]（SR-TC/SC-TC 命名同构）
- [[source-guo-imputation-evaluation]] ↔ [[fence]]（FENCE 引用本评测 v1，raw 核实）、[[loft]]（同组后续工作、基线重叠；LOFT raw PDF 缺失，引用关系未在 raw 核实，页面内如实注明）

## Frontmatter 更新

- pristi.md：last_updated 2026-08-29（原已是当日），source_count 11。
- imputeformer.md：last_updated 2026-08-29（原已是当日），source_count 8。
- std-plm.md：last_updated 2026-08-29，source_count 2，confidence medium。
- 新页面三页：created/last_updated 2026-08-29，source_count 1，confidence medium。

## 并行安全备忘

- 独占文件：本报告与三个新页面；共享文件 pristi/imputeformer/std-plm 仅追加本论文相关段落（Edit 前重读、冲突则重试）。
- wiki/index.md 与 wiki/log.md 在全部其他工作完成后最后追加；未 git add/commit/push；未动 flow matching guide 相关文件；未修改 raw/。

## 审查补记（2026-08-29 lint-fix）

对照 raw/guo-imputation-evaluation-st-traffic-arxiv-2024.pdf 全文复核（pdftotext 默认 + -layout 双抽取，Table III/IV 逐格对位）。修复：警告 1——"TW 上 IGNNK 在 SCTR 下最差"超出论文表述（原文 "However, under the SCTR missing pattern, IGNNK performed poorly"，未称最差），source 页/benchmark 页/traffic-missing-patterns 页改"表现差"。警告 2——"正文 Sec. IV/V 两处写 '10'"计数不准，论文实际三处（Sec. IV "select the following 10 models"、V.B.3 "10 recently proposed sequence imputation and prediction models"、V.B.4 "10 baselines"），本报告与两页已更正。警告 3——pristi.md 评测节 "（PriSTI 对应线性插值先验）" 归因语境错配：top-1/2/3 计数语境（Fig 13 段）论文将 PriSTI 与 BRITS/GCASTN 同归为利用时延机制，"先行线性插值"出自模型设计建议节（Sec V.F.1），已分别注明；并补评测数据集级结论（BRITS 在 Seattle 一致最优、PriSTI 未被列为任一数据集整体最优），使原文/评测两套口径的分立在页面本身显式化。信息级修复：source 页缺失率表述对齐论文口径（低于 0.5 相对稳定、达到 0.7 与 0.9 显著恶化）；benchmark 页数据集相关性句补空间维数字（62.31%/72.62%/8.29%/1.30%）并为主要结论补表/图定位（Fig 7-10、Fig 11/12、Fig 13、Table III/IV、Sec V.F.2/V.F.3）；模型选择建议 "SRTC→GCASTN、SCTR→BRITS" 的模式映射系 wiki 消歧（论文原话为 temporal continuous / spatial continuous 缺失模式，未逐一点名四类中的哪一个），benchmark 页与 traffic-missing-patterns 页改用论文原词并注明。其余核实无误：Table III/IV 全部数字与排名结论、评测范围（11 模型 + LAST × 4 数据集 × 20 场景 × 10-90%）、四分类定义与掩码构造协议、论文三处自身不一致的分立记录、v1 题名经 FENCE 参考文献核实（Guo, S.; Wei, T.; et al. 2024. An Experimental Evaluation of Imputation Models for Spatial-Temporal Traffic Data. arXiv:2412.04733）、PDF 全文无 MCAR/MAR/MNAR/Rubin 术语（与 [[missing-not-at-random]] 分立成立）、LOFT 引用关系未核实表述保持、6 个相关页面脚注三方一致且 source_count 相符（source 1、benchmark 1、missing-patterns 1、pristi 11、imputeformer 8、std-plm 2）。log 中本 ingest 条目按仅追加原则未回改，以上更正以本补记与 lint-fix log 条目为准。
