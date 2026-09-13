# Ingest 报告：Trace as State

日期：2026-09-13。用户入口：https://arxiv.org/html/2609.02702。

主来源：Xu Zou（Z.ai）、Jie Tang（Tsinghua University），*Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers*，arXiv:2609.02702v1，2026-09-02，preprint。完整阅读 21 页 PDF、HTML 正文、Limitations、References 与附录 A–F；用 TeX 核对表格，用原图 PDF 核对图版。[^src-trace-as-state]

## 创建

### 主论文知识页

- `wiki/source-trace-as-state.md` — WHY：给主论文建立唯一来源锚点、版本链接、PDF/TeX 校验和及 420 字摘要；PDF、HTML、TeX 不重复计来源。
- `wiki/trace-as-state.md` — WHY：作为阅读入口，用一个有向图教学例解释为什么晚发现的状态值得前置重读，再连接机制、对照、结果与适用范围。
- `wiki/conditional-state-update.md` — WHY：把可复用的条件状态更新概念独立出来，完整说明残差映射、一般 cut 下界、最坏情况构造、两侧紧界、退化例及有限 KV 状态的口径。
- `wiki/trace-as-state-evidence.md` — WHY：集中保存全部主结果、控制消融、配对区间、难度分箱和图版，避免主要均值、未校正区间与机制解释互相替代。
- `wiki/trace-as-state-reproduction.md` — WHY：实验能否还原取决于模型版本、字段、插入边界、评分与成本口径；集中记录已公开的精确文本、36 行 token 用量、12 组增量和不能凭空补齐的配置。
- `wiki/trace-as-state-positioning.md` — WHY：直接回到四篇关键原作比较重复对象、状态来源、跨遍反馈、内部信号需求，避免把已有重读思想或不同基准结果误写成历史优先权与性能排名。

### 相关论文来源页

- `wiki/source-re-reading-improves-reasoning.md` — WHY：核查 Re2 的完整 Input Query 重复与引导语，解释主论文 Table 3 中 `[x,q,x,q]` 的适配口径。
- `wiki/source-core-context-repetition.md` — WHY：核查 CoRe 的上下文重复与有序子序列保证，区别于模型已经生成的 reasoning trace。
- `wiki/source-state-over-tokens.md` — WHY：说明外化计算状态这一概念来源，并保留它不是独立实证方法、也不保证忠实解释的范围。
- `wiki/source-recontext.md` — WHY：核查内部 attention 信号、原文证据选择及回放的位置，区别白盒证据回放与生成轨迹前置。

### 基准来源页

- `wiki/source-graphwalks-dataset.md` — WHY：保留官方任务定义、prompt 结构、评分示例与数据修复范围；不将数据集卡示例代码当作主论文 runner。
- `wiki/source-mrcr-dataset.md` — WHY：保留多实例绑定、随机前缀与 SequenceMatcher 的一手定义，并明确数据集卡没有指明论文所谓 MRCRv2 对应哪个快照。
- `wiki/source-nub-1m-benchmark.md` — WHY：保留 Season 2、二十题、当季答案公开策略及单次排行榜口径，区别论文的五次重复评估。

合计 **13 个新 wiki 页面**：8 个 `source-summary`、1 个 `concept`、1 个 `technique`、3 个 `analysis`。

### 原始材料与图版

- `downloads/trace-as-state.pdf`、`downloads/trace-as-state.tar.gz`、`downloads/trace-as-state.txt` — 主论文版本化原文、TeX 包与完整 `pdftotext -layout` 抽取。
- `downloads/re-reading-improves-reasoning.pdf`、`downloads/core-context-repetition.pdf`、`downloads/state-over-tokens.pdf`、`downloads/recontext.pdf`，以及同名 `.txt` — 四篇相关原作与文本抽取；各来源页分别标明著录、链接和阅读范围。
- `downloads/graphwalks-dataset.md`、`downloads/mrcr-dataset.md`、`downloads/nub-1m-benchmark.md` — 三套基准资料的内容快照。
- `wiki/assets/trace-as-state/` 中 12 个 PNG — 原论文 Figure 1–6 的完整面板集合，由 TeX 包内 12 份原图 PDF 转换；保留原内容，在主来源页记录 CC BY 4.0 许可与转换说明。

`raw/` 是只读目录，本次没有向其中新增、修改或删除任何文件；也没有修改 `.obsidian/`。在线输入的原文保存在既有 `downloads/`，其文件 basename 与本次来源 slug 对应；没有制造并不存在的 `raw/` 路径。

## 修改

- `wiki/context-window-extension.md` — WHY：分开“窗口能装多长”和“读材料时有哪些状态已可用”；新增主来源，`source_count` 3→4。
- `wiki/kv-cache-compression.md` — WHY：区分 KV 淘汰/压缩与跨遍条件化重读，澄清有限状态抽象包括全部 KV 和缓冲；1→2。
- `wiki/long-context-scaling-gap.md` — WHY：把 Attention Dilution 的长度变量与本论文的顺序变量分开，不把新结果写成既有缩放问题被解决；1→2。
- `wiki/in-context-learning.md` — WHY：区分其他题目的 demonstrations 与同一道题的首轮推理文本；1→2。
- `wiki/non-parametric-policy-memory.md` — WHY：区分 JitRL 奖励经验对 logits 的作用与本论文文本状态对下一遍输入处理的作用；1→2。
- `wiki/prompt-as-prefix.md` — WHY：区分预先构造的任务/统计提示与读过同题后产生的推理状态；1→2。
- `wiki/structural-token-sorting.md` — WHY：区分 lead–lag 图上的输入 token 排序与跨遍轨迹前置，不把物理依赖与状态发现顺序等同；1→2。
- `wiki/index.md` — WHY：登记全部 13 个新页面，复用原有分类与 `(continued)` 形式；新增条目在各类别内按 slug 排序，更新日期并补齐索引元信息。
- `wiki/log.md` — WHY：仅在末尾追加本次 ingest、来源边界、创建/修改清单与实际校验结果，不改变任何旧记录。

七个主题页均更新 `last_updated`。它们的原有结论未因新论文被无依据地整体替换。

## 新建交叉链接

- [[trace-as-state]] ↔ [[source-trace-as-state]]、[[conditional-state-update]]、[[trace-as-state-evidence]]、[[trace-as-state-positioning]]、[[trace-as-state-reproduction]]。
- [[trace-as-state]] ↔ [[context-window-extension]]、[[kv-cache-compression]]、[[long-context-scaling-gap]]、[[in-context-learning]]、[[non-parametric-policy-memory]]、[[prompt-as-prefix]]、[[structural-token-sorting]]。
- [[trace-as-state-positioning]] ↔ [[source-re-reading-improves-reasoning]]、[[source-core-context-repetition]]、[[source-state-over-tokens]]、[[source-recontext]]。
- [[trace-as-state-reproduction]] ↔ [[source-graphwalks-dataset]]、[[source-mrcr-dataset]]、[[source-nub-1m-benchmark]]。

上述双向关系已通过实际文件中的 wikilinks 核对；13 个新页均有来自其他内容页的入链，不依靠自身脚注或索引制造非孤立状态。

## 论文内容覆盖

| 原文范围 | 入库位置与处理 |
|---|---|
| §1 问题、贡献与 Figure 1 | 主入口；区分后续 token 仍可访问前文与早期表示无法被同遍后文改写 |
| §2 Related Work | 定位页；四篇关键原作直接核查，其余相关工作明确仅按主论文转述，不伪称逐一核查 |
| §3 / Table 1 / Eq. 1–5 | 主入口与条件状态页；完整符号、状态路径、残差映射、serializer 与两种顺序 |
| §4.1 / Table 1-1、1-2 | 复现页；版本、供应商、预算、effort、题数、重复次数与异常评分 |
| §4.2 / Table 2 | 证据页；全部 27 个模型—任务—指标格的三条件值，共 81 个数字 |
| §4.3 / Table 3 | 证据页；十个条件、四个指标，共 40 个数字与控制边界 |
| §4.4 / Figure 2 | 证据页；嵌套轨迹数设置、图版与端点；中间点无精确标签，不从像素估数 |
| §5 与 Limitations | 主入口、复现页；接口、成本、覆盖范围与尚未实验的扩展方向 |
| Appendix A | 条件状态页；一般下界、全自映射构造、两个 cut、紧性、m=1、m=4、完整工作状态 |
| Appendix B | 复现/证据页；空集、末行语法、MRCR 前缀、NUB judge、Question First 的 182 个资格性零分 |
| Appendix C / Table 4 | 复现页；已公开提示原文与五条轨迹的结构转写，精确插入边界与未给出的文本分开 |
| Appendix D / Figure 3–4 | 证据页；全部分箱、题数、132 个图内数值及事后选区限制 |
| Appendix E / Table 5 / Figure 5–6 | 证据页；24 格配对差与区间、20,000 次 problem-cluster bootstrap、边际/配对区间区别 |
| Appendix F / Table 6 | 复现页；全部 36 行聚合 token、180 个数字、12 组位置对照增量与定价公式 |

## 矛盾与证据边界

- **没有直接来源冲突需要将旧页设为 superseded/disputed。** 长度退化、KV 容量、奖励记忆、token 排序与本论文的状态可用时机不是同一个实验问题；新增关联以 `[INFERENCE]` 或结构导航标明。
- **理论不是实测显存承诺。** 指数差异来自确定性、精确、单遍任务的对抗性存在构造，原文明确不把它视作现实长上下文任务的字面模型。[^src-trace-as-state]
- **26/27 不等于 26 项显著。** 它统计均值；Table 5 的分母为 24，20 个未校正 95% 区间完全大于零，四个跨零；NUB 不在该配对表。[^src-trace-as-state]
- **Re2 的适配不制造错误的优先权争论。** 原作重复整体 Input Query；主论文把长上下文与问题按 `[x,q,x,q]` 呈现，可视为长文适配。原作没有报告本论文的新基准数字，而主论文未给出全部消融引导语，不能据此假定实现文本完全相同。
- **分隔符和截断以公开细节分立记录。** 正文举例的 `<trace_start>`/`<trace_end>` 与 Appendix C 的固定标签不同；GraphWalks preamble 提到 tail windows，但实际复用规则截取每条轨迹前 50,000 字符。没有改写原始来源替作者统一措辞。[^src-trace-as-state]
- **输出预算不是传入的输出上限。** 作者没有显式设置 custom maximum-output；provider 实际默认值与全部 solver 采样配置仍缺失。[^src-trace-as-state]
- **数据集卡不是实验实现。** GraphWalks 的 F1 示例、MRCR 的文本说明与代码细节、基准修复 commit 的范围、NUB 单次排行榜与未公开的当季答案，均与主论文实验口径分开；详见复现页及三个来源页。
- **公开材料范围有限。** 所查 arXiv 页与 33 个普通文件的 TeX 包中未见 runner 或逐题 records；不把这一观察扩大成“全网不存在实现”。Table 6 的 Qwen MRCRv2 512K `†` 未见解释，保留该未说明状态。
- **不补齐缺失数字。** Figure 2 中间点没有数值标签；NUB prompt/judge 全文、MRCR 的消息级插入细节、完整解码参数、逐调用计费记录缺失，均明确列出。

## 已执行的交付校验

### 转录与计算

- 主结果：**81/81** 数字与 TeX 一致。
- 控制消融：**40/40** 数字与 TeX 一致。
- 配对差与置信区间：**72/72** 数字与 TeX 一致；重新计数得到 20 个完全正区间、4 个跨零区间。
- Token 表：**36/36 行、180/180** 数字与 TeX 一致；`Total = First Pass + 第二遍` 在独立舍入后最大分量差为 0.001M，未改动原表数值。
- Figure 3/4：**132/132** 带符号数值与 PDF 文本标签一致；按原题数回算 12 个剖面，最大差 **0.05 个百分点**。
- Table 2 重新计数得到前置相对后置 **26 正、1 负**；相对 First Pass **27 正**。
- 最坏情况教学例：穷举 4 状态的 **256** 个自映射，检查 **32,640** 对不同映射均可被某个条件区分，核实先给条件 2 比特/后给条件 8 比特；检查单状态零比特与循环移位受限族 `m=1…8`。
- 主入口自建图例：运行 BFS，第一层为 B/C，第二层为 D/E；明确它不是论文的真实模型输出。

### 文档契约与导航

- **20/20 个受影响内容页**通过必需 YAML 字段、日期、type/status/confidence、脚注定义、唯一来源计数与出链目标检查。
- 八篇 source-summary 的摘要正文字数（扣除引用标记、Markdown 修饰与空白；英文字符及数字计入）：主论文 420、Re2 436、CoRe 444、SoT 377、ReContext 462、GraphWalks 434、MRCR 429、NUB 454；均在 300–500。
- **13/13 个新增页**在索引中各登记一次，新增条目在各类别内按字母顺序排列；索引历史正文保持原样。
- **13/13 个新增页**有其他内容页入链；七个旧主题页与主入口之间的链接均为双向。
- **12/12 个图像嵌入**指向存在且具有正确 PNG 文件签名的资产；原图来源与许可已记录。
- 日志追加前的 **571,756 字节**逐字节保留，只在其后追加本次记录。

这是对资料、算术、链接与元信息的验证，**不是对三款商业模型的独立基准复现**，也不是全仓库历史 wiki lint。

## 有意保留的既有约定

- 索引已有大量 `(continued)` 分类；本次复用并只对新增组排序，没有以 ingest 为由重排整个历史索引。
- `wiki/log.md` 原本没有 frontmatter，且具有更具体的“仅追加”约束；本次不改写其头部或历史记录，新记录以日期标识。
- 既有主题页保留原 `created` 与原结论；更新 `last_updated` 和实际来源计数。来源载体多份不等于独立实验证据，因此新内容没有机械升级为 `confidence: high`。

[^src-trace-as-state]: [[source-trace-as-state]]

## 收尾

完成校验后，删除本次生成且未被正文引用的 `downloads/trace-as-state-assets/` 图版解包中间目录。14 份原文、文本与基准快照及 12 个 PNG 面板全部保留；再次核对主论文 PDF/TeX 的 SHA256，与来源页记录一致。没有留下实验 runner、未完成的验证标记或待填写配置。
