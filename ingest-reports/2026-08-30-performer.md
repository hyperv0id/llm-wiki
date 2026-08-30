# Ingest 报告：Rethinking Attention with Performers（Choromanski et al., ICLR 2021）

- raw 文件：`raw/performer-choromanski-2020.pdf`（arXiv:2009.14794v4 [cs.LG] 19 Nov 2022；每页页眉 "Published as a conference paper at ICLR 2021"，会议标识在 PDF 内核实）
- 处理日期：2026-08-30

## 创建

- wiki/performer.md — WHY：Performer 在 [[wire]]、[[source-bigst]]、[[urbanpg]]、[[graphgps]] 等多页被按名引用但无专页（lint「正文提及但无页面」项）；用户设计分析中 PRF 线性注意力的来源，需要以 PDF 原文为准的机制、定理与实验页（含 FAVOR+ 两组件、命名细节 Performer vs Performer-SOFTMAX、论文自述局限）。
- wiki/positive-random-features.md — WHY：论文引入的独特机制概念（PRF 正随机特征无偏 softmax 核估计、trig vs 正特征 MSE 闭式、ORF 方差控制、SMREG 下界），BigST 系/UrbanPG/MAGE 多页以 "PRF" 指称该机制，单独立页避免 performer.md 过载并承接全 wiki 的 PRF 链接；含 ≥1 内联引用。
- wiki/source-performer.md — WHY：每个 raw 文件一页 source-summary 的工作流规定；368 汉字在 300-500 区间，注明版式核实与 raw 文件名。

## 修改

- wiki/long-sequence-feature-extractor.md — WHY：正文「Performer PRF 近似 softmax 核」是本 wiki 与 Performer 机制最直接的接口；补 [^src-performer]（PRF 无偏估计器出处）+ [[performer]]/[[positive-random-features]] 链接，source_count 1→2。
- wiki/linearized-spatial-convolution.md — WHY：LSC 用 PRF 分解自适应邻接，同一接口；补引用与链接，source_count 1→2。
- wiki/bigst.md — WHY：实体页正文提及 Performer PRF，补 wikilink（机制断言仍归 [^src-bigst]，不重复归因）；关联表补机制来源。
- wiki/source-bigst.md — WHY：同上，方法节补 [[performer]] 链接（source-summary 仅链接、不加新源，保持 source_count: 1）。
- wiki/wire.md — WHY：两处按名提及 Performer（线性注意力兼容、GraphGPS+ReLU Performer 实验骨干），补首现链接与相关页条目（结构性，source_count 不变）。
- wiki/linear-attention-unified-framework.md — WHY：该框架覆盖「核随机特征」路线但缺 Performer 页链接；补 [[performer]]/[[positive-random-features]] 条目及机制句引用，source_count 2→3。
- wiki/urbanpg.md — WHY：STCA 的 φ 来自 Performers 随机特征映射，补链接与相关页条目；页面保留 UrbanPG 论文「sin/cos 编码」的原始转述，与 Performer 论文「正特征替代 sin/cos」的差异在 PRF 页按各自来源归因说明，不构成矛盾。
- wiki/spectral-kernel-linear-attention.md — WHY：分析页提到「可与 Performer 结合」，补链接与相关页条目。
- wiki/graphgps.md — WHY：Performer 是论文实例化的 GlobalAttn 选项且消融中被评估，补首现链接与相关页条目（纯结构性）。
- wiki/source-graphgps.md — WHY：方法句提及 Performer，补 wikilink。
- wiki/mage.md — WHY：关联条目「用 PRF 近似」补 PRF 机制页链接。
- wiki/linear-adaptive-graph-learning.md — WHY：动机句提及 BigST 的 PRF 近似，补链接。
- wiki/ragc.md — WHY：关联条目「PRF 核近似」补链接。

## 未修改（逐页判断）

- wiki/stg-attention.md — 用户指定只读不改（未提交页面，属并行代理范围）。
- wiki/traffic-forecasting.md、wiki/gwnet.md、wiki/large-scale-spatial-temporal-graph.md — 均有 PRF/Performer 提及，但属并行代理未提交改动范围（并行安全规则禁改），链接补全留待后续 lint。

## 矛盾核对

- grep 全 wiki 无既有 Performer 机制论断与 PDF 冲突：[[graphgps]] 的「Performer 性能低于全秩 Transformer 但可扩展」、[[wire]] 的实验数字均为对各自源论文的转述，归因正确；[[urbanpg]] 的「sin/cos 编码」为对 UrbanPG 论文的转述，与 Performer 论文主张正特征的差异属转述口径而非事实矛盾，未触发矛盾解决策略任何情形，无 status 变更、无争议章节。

## 新建交叉链接

- [[performer]] ↔ [[positive-random-features]]（机制页 ↔ 使用它的架构页）
- [[performer]] ↔ [[wire]]（WIRE 线性注意力兼容声明 + ReLU Performer 实验骨干）
- [[performer]] ↔ [[graphgps]]（Performer 为可替换 GlobalAttn 实例 + 消融）
- [[performer]] ↔ [[linear-attention-unified-framework]]（核随机特征路线成员）
- [[performer]] ↔ [[long-sequence-feature-extractor]]、[[performer]] ↔ [[linearized-spatial-convolution]]（BigST 时间/空间维 PRF 线性化）
- [[performer]] ↔ [[urbanpg]]（STCA 的随机特征映射来源）
- [[performer]] ↔ [[spectral-kernel-linear-attention]]（旋转=随机特征的图核读法）
- [[positive-random-features]] ↔ [[mage]]、[[positive-random-features]] ↔ [[linear-adaptive-graph-learning]]、[[positive-random-features]] ↔ [[ragc]]（PRF 链接补全）
- [[source-performer]] ↔ [[performer]]、[[source-performer]] ↔ [[positive-random-features]]（脚注对）

## 引用约定

- 新 slug：`[^src-performer]` → `[[source-performer]]`（raw/performer-choromanski-2020.pdf 去扩展名）。
- frontmatter 更新：所有修改页 last_updated: 2026-08-30；source_count 按唯一源数调整（linear-attention-unified-framework 3、long-sequence-feature-extractor 2、linearized-spatial-convolution 2，其余不变）；新页 confidence: medium（单一源文件、无反驳）。
