---
name: wiki-ingest
description: "把一个 raw/ 源文件写进 llm-wiki：读全文、更新既有实体/概念/技术页、新建有引用的页、写 source-summary、维护 index/log 与 WHY 报告。仅在用户明确要求 ingest 时使用（/skill:wiki-ingest、dashboard ingest 按钮、点名某个 raw 文件入库）；提问、闲聊、讨论不触发。"
disable-model-invocation: true
user_invocable: true
---

# wiki-ingest：单源摄入

用户要求把一份 `raw/` 源文件写进 wiki。页面 schema、frontmatter 字段、引用格式、文体与证据等级在根目录 `AGENTS.md`，动手前先读它；本技能只管流程。

## 硬约束

- `raw/` 不可变，只读。判断源文件有错时在 `wiki/` 写更正页，不改源文件。
- 新建的 entity / concept / technique 页至少含 1 个 `[^src-*]` 引用，否则不建。
- 不给源文件没写的内容补数字、因果或优越性判断。源是 PDF/HTML 转换、表格缺列、图内数值不可读时，如实写进 WHY 报告的「数值备注」。

## 流程

1. **读源文件全文**。`raw/*.pdf` 先 `pdftotext -layout` 转文本；已是 md 则直读。
   - 论文类长文交子 agent 读，同时最多 2 个。主 agent 只收它们回报的结构化结论（问题 / 机制 / 证据 / 边界），不重读全文。
2. **读 `wiki/index.md`**，列出该源涉及的已有 entity / concept / technique 页面。
3. **逐页判断**：补新信息并加引用；给既有论断补第二条引用；发现冲突按「矛盾处理」。
4. **新建页面**：只在源提供至少一条可引用论断时建。文件名小写、连字符、无空格。涉及模型结构的页面，前向数据流按「数据流与维度变换」一节写，不用张量表。
5. **写 source-summary**：`wiki/source-<slug>.md`，`type: source-summary`，300–500 字，覆盖核心论点、贡献、局限；`source_count` 与正文实际引用数一致。
6. **回打数值**：正文关键数字回原文核对，记录表号 / 图号 / 章节。原文表格与正文不一致时两处并列报告，不替论文挑一个更好看的数字。
7. **更新 `wiki/index.md`**：按 frontmatter `type` 分五段（Sources / Entities / Concepts / Techniques / Analyses），段内字母序，每页仅一条。
8. **更新 frontmatter**：所有被改页面刷新 `last_updated`，重算 `source_count`，调整 `confidence`。
9. **写 `ingest-reports/YYYY-MM-DD-<slug>.md`**（格式见下）。
10. **追加 `wiki/log.md`**：

   ```
   ## [YYYY-MM-DD] ingest | {标题}
   {1 段：源解决了什么问题、方法机制、关键证据与数字；有边界或未核实处直接写}
   创建的页面：[[page1]], [[page2]]
   更新的页面：[[page3]]（一行改动摘要，source_count 1→2）
   ingest 报告：[[ingest-reports/YYYY-MM-DD-slug|WHY 报告]]
   数值备注：{源转换缺陷 / 表图不可读 / 正文与表格冲突}
   ```

11. **提交**：dashboard 路径由 `server.py` 自动 commit；手工路径以 `ingest <标题>` 作为 commit message 提交 `raw/`、`wiki/`、`ingest-reports/`。

## 数据流与维度变换（模型类页面必用）

讲模型前向传播时，**禁止用表格罗列张量清单或逐步形状变化**——表格是线性的，表达不了并行支路、汇合过程，也看不出矩阵乘法消掉了哪一维。改用「分阶段计算图块（staged computation blocks）」。

### 1. 维度符号速查

「数据流」小节开头用一到三行交代超参的物理意义，不列中间张量的大表：

> **维度符号**：$N$ 序列长度（token 数）；$d$ 单头通道维；$m$ landmark 数；$k$ 每个专家选中的 token 数。

形状统一记作 $[\text{行}, \text{列}]$，与论文记号一致。论文按单头书写、本页按多头展开时，推断部分标 `[INFERENCE]`。

### 2. 按前向拓扑分阶段

- 一个阶段一个小标题，编号加动词短语：`### Stage 1 · 造 landmark query`。
- 并行支路在同一阶段内用无序列表拆开，各占一个公式块：`**支路 A：共享专家（常开，宽 $m$）**`、`**支路 B：形变专家（稀疏，宽 $k$）**`。支路前先交代它读哪些张量、形状是什么。
- 支路汇合单独成阶段，写清合并方式（拼接、按打分路由、online softmax 合并等）。

### 3. 每个公式块的四要素

```latex
$$\begin{aligned}
S^{kv} &= K^{\top}\tilde{Q}/\sqrt{d} && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, m]\to[N, m]\ \text{——消去 } d\text{，每列是一个 landmark 对全部 } N \text{ 个 key 的打分}\\
\mathcal{I}_i &= \operatorname{Top}_k\!\left(K^{\top}\tilde{q}_i\right) && \text{// } [N, m] \xrightarrow{\text{沿 } N \text{ 维逐列 Top-}k} [k, m]\ \text{——每组 } k \text{ 个 key 下标（取值 } 1..N\text{）}
\end{aligned}$$
```

左侧放张量符号与公式，右侧 `&& \text{// }` 注解必须同时给出**维度变换**和**一句话物理含义**；多行用 `\\` 对齐，注解内部不换行，物理含义用 `\ \text{——}` 接在维度变换之后。

### 4. 维度标注定式

| 操作 | 写法 |
|---|---|
| 矩阵乘法 | 用 `\bcancel{}` 标出被消去的维度：`[N, \bcancel{d}]\times[\bcancel{d}, m]\to[N, m]` |
| 规约 / 池化 / 采样（softmax、Top-$k$、ArgMax、AvgPool） | 带文字的箭头写清沿哪一维操作：`[N, m] \xrightarrow{\text{沿 } N \text{ 维逐列 Top-}k} [k, m]` |
| 索引 / 重排（Gather、ArgSort） | 标明索引张量的形状如何作用于原张量：`[d, N] \xrightarrow{\text{按 } \mathcal{I}\in[mk] \text{ 索引列}} [d, mk]` |

### 5. 表格的用途

表格只承载论文实验数值（数据集 × 指标 × 对照 × 出处），不承载张量形状。已完成该格式的页面可参照 `wiki/mita-attention.md` 的「数据流与维度」小节。

## 矛盾处理

新源与既有论断冲突时，按冲突性质三选一，不得静默覆盖：

- **新源更新且证据强**（多源支持、时间更近、方法更直接）：旧论断移入 `## 历史论断`，正文写新论断，注明日期与取代关系。
- **日期相近或新源证据弱**（单一源、 preprint、口径不明）：正文建 `## 争议`，并列两种论断，页面 `status: disputed`，写清待解决点。
- **新源明确反驳旧源**：旧 source-summary 页设 `status: superseded` 与 `superseded_by: [[新页]]`。

三种情况都追加 `wiki/log.md`：

```
## [YYYY-MM-DD] contradiction | {页面} | {情况 1/2/3}
{旧论断} vs {新论断}。解决方案：{所选情况}。
```

## 文案压实（可选）

页面初稿完成后可交外部模型压缩，判据是信息量不变：

- `omp -p --model google/gemini-3.8-flash:high`，无工具、无会话，只传页面正文与文体规则。
- 目标：字符数降 10%–20%。只合并同义句、删无信息量的连接词与冗余指代、把「对……进行……」拆成动词。
- 逐字不动：数字、单位、表/图/节编号、方法名、数据集名、作者名、venue、LaTeX 公式、`aligned` 维度块与其中的 `\text{//}` 注解、引用标记、wikilink、markdown 表格、mermaid 块、YAML frontmatter。
- 不加不删事实、判断、因果解释、引用与限定语；归属标记（作者报告 / 论文自述 / 未实测）保留；断言强度不上调也不下调。
- 保留原 heading 层级与段落顺序，段落不拆成列表。
- 输出若滑向文言（「遂、亦、须、皆」）或评价性措辞（「巧妙地、收益可观」），整轮弃用重跑，并在 log 里记回退项。

## WHY 报告格式

```markdown
# Ingest 报告：{source_name}

## 创建
- wiki/page.md — WHY：一行理由

## 修改
- wiki/page.md — WHY：一行理由

## 新建交叉链接
- [[a]] ↔ [[b]]

## 数值备注
- {源缺陷、表图不可读、正文与表格冲突；无则写「无」}
```

## 完成判据

- 该源的每条主要论断在 wiki 里都有落点，且带 `[^src-*]` 与脚注定义。
- 模型类页面的前向数据流是分阶段 `aligned` 块（维度变换 + 物理含义），不是张量形状表。
- `wiki/index.md`、`wiki/log.md`、`ingest-reports/`、被改页面 frontmatter 均已更新。
- WHY 报告里的每条创建/修改都能对上一个具体页面。
