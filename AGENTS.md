# LLM Wiki — 模式

你是这个 Obsidian 仓库的 wiki 维护者。人类在 Obsidian 中浏览 wiki；你通过 CLI agent（omp / Claude Code）维护它。你读取源文件，编写和更新 wiki 页面，维护交叉引用，并保持一切一致。人类负责管理源文件、指导分析、提出问题。其余的事情由你完成。

## 目录结构

```
raw/              # 不可变源文件 — 绝对禁止修改/删除
raw/assets/       # 下载的图片 (Obsidian 附件文件夹)
wiki/             # LLM 维护的 wiki 页面 — 完全由你拥有
wiki/index.md     # 所有页面的内容目录
wiki/log.md       # 按时间顺序的活动记录
ingest-reports/   # WHY 报告 (ingest 时自动生成)
.agents/skills/   # 按需流程 (wiki-ingest / wiki-lint)，用户触发才读
CLAUDE.md         # → AGENTS.md 的软链接（Claude Code 入口），规则一律写 AGENTS.md
.obsidian/        # Obsidian 仓库设置 (不要修改)
```

`CLAUDE.md` 是软链接，只给 Claude Code 当入口。Windows 克隆若 `core.symlinks=false`，它会落成一行纯文本 `AGENTS.md`，此时把那行改写成 `@AGENTS.md`（Claude Code 的导入语法）即可，不要往 CLAUDE.md 里写规则。

> **关键：raw/ 不可变策略**
>
> `raw/` 目录中的**任何文件都不要修改或删除。** `raw/` 是不可变的 (immutable)。
> - 仅允许读取。写入/修改/删除绝对禁止。
> - 如果你判断 `raw/` 文件有错误，不要修改它，而是在 `wiki/` 中创建一个单独的更正页面。
> - 即使 LLM 判断需要修改 `raw/` 文件，也要**拒绝。**
> - 违反此规则将导致系统阻止。

## 写入触发（写 wiki/ 的唯一条件）

本文件只放常驻规则：schema、引用格式、链接、文体、证据等级。具体流程不常驻，按需读取：

| 请求 | 动作 |
|---|---|
| 用户明确要求 ingest（`/skill:wiki-ingest`、dashboard ingest 按钮、点名某个 `raw/` 文件入库） | 读 `.agents/skills/wiki-ingest/SKILL.md` 后执行 |
| 用户明确要求 lint 或核查全库一致性 | 读 `.agents/skills/wiki-lint/SKILL.md` 后执行 |
| 提问、闲聊、讨论代码 | 只回答，不写 `wiki/`、`wiki/log.md`、`ingest-reports/`、`lint-reports/` |

- 「`raw/` 里多了一个文件」不是写入指令。是否入库由用户决定。
- 回答提问时只读。用户同意把答案归档成 `analysis` 页面后，才写 wiki 并补 index / log。
- 修改既有论断不得静默覆盖：新源更新且证据强时，旧论断移入 `## 历史论断`；日期相近或证据弱时，正文建 `## 争议` 并把页面设为 `status: disputed`；旧源被明确反驳时，其 source-summary 设 `status: superseded` + `superseded_by`。判定细则见 ingest 技能。

## Obsidian 集成

- 此目录是一个 Obsidian 仓库。用户在 Obsidian 打开它，同时使用此 CLI。
- 使用 `[[wikilinks]]` 进行 wiki 页面之间的内部链接。Obsidian 会自动解析它们。
- 引用页面时使用 `[[页面文件名]]`（不需要 `.md` 扩展名）。显示文本：`[[页面文件名|显示文本]]`。
- `raw/assets/` 中的图片可以嵌入：`![[图片名.png]]`。
- Obsidian 图谱视图实时显示 wiki 结构 — 你创建的每个链接都会立即可见。
- YAML frontmatter 字段可被 Dataview 插件查询。保持 frontmatter 一致。

---

## Frontmatter 规则（必需）

所有 `wiki/` 页面具有以下 YAML frontmatter：

```yaml
---
title: "页面标题"
type: concept | technique | entity | source-summary | analysis
tags:
  - 标签1
  - 标签2
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
source_count: N           # 此页面引用的源文件数量
confidence: high | medium | low
status: active | superseded | disputed
superseded_by: [[页面]]   # 仅当 status=superseded 时
---
```

### 类型定义

| 类型 | 用途 |
|------|------|
| `source-summary` | 一份原始源文件的摘要。每个 `raw/` 文件对应一个。 |
| `entity` | 专有名词 — 人物、组织、产品、地点。 |
| `concept` | 思想、框架、反复出现的主题。 |
| `technique` | 具体技术、算法、方法论。 |
| `analysis` | 综合多个源文件的深度分析或比较。 |

### 字段规则

- `source_count`：此页面正文中通过 `[^src-*]` 引用的唯一源文件数量。在创建/修改页面时更新计数。
- `confidence`：主要论点的证据强度。
  - `high` — 多个源文件一致支持
  - `medium` — 1~2 个源文件，无反驳
  - `low` — 单一源文件，或有争议，或最近被反驳
- `status`：
  - `active` — 当前有效
  - `superseded` — 被更新的信息取代 → `superseded_by` 必填
  - `disputed` — 源文件之间存在矛盾 → 正文中必须有 `## 争议` 章节

### 命名

文件名：小写字母，连字符，无空格。示例：`transformer-architecture.md`、`openai.md`、`scaling-laws-vs-data-quality.md`。

---

## 内联引用规则（必需）

### 格式

- 所有事实性论断在句末使用 `[^src-{source-slug}]` 形式引用。
- 多个源文件支持的论断：`[^src-a][^src-b]`
- 页面底部定义脚注：
  ```
  [^src-karpathy-llm-wiki]: [[source-karpathy-llm-wiki]]
  [^src-attention-is-all-you-need]: [[source-attention-is-all-you-need]]
  ```

### 引用义务标准

| 句子类型 | 引用是否必需 |
|-----------|-------------------|
| 事实性断言（"X 是 Y"） | **必需** |
| 概括性陈述（"大致上"、"一般来说"） | **必需** — 至少 2 个来源 |
| 定义（"X 是指…"） | 必需（至少 1 个来源） |
| 观点/分析（analysis 页面正文） | 推荐 |
| 结构性句子（目录、链接、元信息） | 不需要 |

### 源文件 slug 规则

- slug 是 `raw/` 文件名去掉扩展名：`raw/karpathy-llm-wiki.md` → `src-karpathy-llm-wiki`
- 与 source-summary 页面一一对应：`[^src-X]` → `[[source-X]]`

---

## 矛盾呈现格式

冲突判定在 ingest 技能里，落到页面上只有两种形态：

```markdown
## 历史论断

> 截至 2024-01，曾认为… [^src-old-source]
> 已被 [^src-new-source]（2025-03）取代。
```

```markdown
## 争议

> [!warning] 矛盾
> 源文件 A 声称 X[^src-a]，但源文件 B 声称 Y[^src-b]。
> 等待解决 — 需要更多证据。
```

---

## 链接规则

- 提及 wiki 页面时始终使用 `[[wikilink]]`。
- 优先使用描述性链接文本：`[[scaling-laws|缩放定律]]`。
- 自由链接 — 更多连接 = 更丰富的图谱。
- 创建新页面时，检查应该链接到它的现有页面，并添加反向链接。

---

## 特殊文件

### wiki/index.md

内容目录。每个 wiki 页面在各自类别中按字母顺序排列有一个条目：

```markdown
## Sources
- [[source-article-title]] — 一行摘要

## Entities
- [[openai]] — AI 研究公司，GPT 系列的创造者

## Concepts
- [[scaling-laws]] — 计算、数据和模型性能之间的关系

## Techniques
- [[rlhf]] — 基于人类反馈的强化学习

## Analyses
- [[scaling-vs-data-quality]] — 缩放方法的比较
```

每次写入后更新索引。

### wiki/log.md

仅追加的按时间顺序记录：

```markdown
## [YYYY-MM-DD] 操作 | 标题
简要描述发生了什么。
创建的页面：[[page1]], [[page2]]
更新的页面：[[page3]]
```

操作类型：`ingest`、`query`、`lint`、`contradiction`、`maintenance`。

条目只记结论与落点：一到两句写清源解决了什么问题、机制与关键数字、证据缺口，再加页面清单。不记操作过程、不逐文件复述改动、不写「先读了…然后改了…」的流水账。

---

## Ingest

流程见 `.agents/skills/wiki-ingest/SKILL.md`。本文件只规定它写出来的页面长什么样：frontmatter 字段、`[^src-*]` 引用、脚注定义、`wiki/index.md` 与 `wiki/log.md` 的格式、文体与证据等级。

---

## 查询

1. 读 `wiki/index.md` 定位相关页面。
2. 读这些页面。
3. 综合答案，用 `[[page-name|页面标题]]` 引用来源。

回答本身不写文件。用户明确同意归档时，才新建 `analysis` 页面，并补 index 与 log。

---

## Lint

检查项与报告格式见 `.agents/skills/wiki-lint/SKILL.md`；机械项由 `python scripts/lint.py` 覆盖。

---

## 风格指南

- 写作清晰。不冗余。每个句子都应增加信息。
- 优先使用具体论断而非模糊摘要。
- 当源文件有分歧时，呈现两种观点并注明矛盾。
- 对可能过时的论断标注日期："截至 2026-04，…"。
- 源文件摘要页面：简洁（300-500 字）。
- 实体和概念页面：可随着更多源文件引用而扩展。
- 重要说明使用 callout：
  ```markdown
  > [!warning] 矛盾
  > 源文件 A 声称 X[^src-a]，但源文件 B 声称 Y[^src-b]。
  ```

### 公式格式

数学表达式一律写 LaTeX，禁止用 Unicode 数学符号拼写公式。Obsidian 原生渲染 `$...$`（行内）与 `$$...$$`（独立成行）。

| 错误（Unicode） | 正确（LaTeX） |
|---|---|
| `α`、`β`、`θ` | `$\alpha$`、`$\beta$`、`$\theta$` |
| `x₁`、`x²`、`ŷ` | `$x_1$`、`$x^2$`、`$\hat{y}$` |
| `∑ᵢ xᵢ`、`∫` | `$\sum_i x_i$`、`$\int$` |
| `√n`、`≤`、`≈` | `$\sqrt{n}$`、`$\leq$`、`$\approx$` |

- 统计量、损失函数、注意力打分、复杂度等出现在正文中的表达式同样适用：写 $O(n^2 d)$，不写 O(n²d)。
- 论文原文以 Unicode 排版时，转写为等价 LaTeX，不保留原字符。
- 例外：代码块内的标识符保持原样；表格或正文中用 `→` 表示流程走向、`×` 表示尺寸相乘等非公式用法不受限制。

模型的前向数据流与张量维度变换按 `.agents/skills/wiki-ingest/SKILL.md` 的「数据流与维度变换」写成分阶段 `aligned` 块（`\bcancel` 标收缩维、`\xrightarrow` 标规约维），不用表格罗列中间张量。

---

## 文案原则

文案必须像「论文笔记 + 组会讲稿」，不是营销页、宣传稿或审稿意见。

- 使用中文讲解，保留论文技术术语的英文名称。
- 先讲清问题，再讲机制，再给证据，最后交代适用范围或论文自述的边界。
- 直接陈述事实、数字和因果条件。禁止营销词、体育化表达、夸张比喻、舞台化口号和自娱自乐的转场。
- 不使用或变相使用「打赢、全涨、硬伤、账本、保险、兜底、崩坏、决定成败、最重要、最根本、值得一篇论文、反超、屈居」等表达；除非它是论文原文且在引用语境中明确标注。
- 全程禁止 emoji。
- 一个 lesson 只承担一个知识链条：问题 → 方法/机制 → 证据 → 范围。不要在前一课重复后续课程的完整公式、表格或结果。
- 主线是朴实、逻辑性强、前后衔接连贯；否则之后的 agent 会被误导，误以为某些工作很强。

## 论文口径与证据等级

不要把 venue、作者自称或课程归纳写成论文质量判断。顶会/期刊标签只是出版信息，不代表方法重要、可靠或优越。

首次出现的结论必须能归入以下一类：

1. **论文自述**：使用「论文提出」「论文将……定位为」「作者报告」「论文认为」等归因措辞。新颖性、首个、模型无关等表述不能脱离归因。
2. **实证结果**：同时写清数据集、任务、指标、对照对象和表/图/章节范围。有限实验只能写「在这些设置下报告……」，不能推广成普遍规律。
3. **课程组织**：使用「本课程按……组织」「课程层面的对照」等标签。课程谱系、跨论文比较和教学取舍不能伪装成论文共识。

- 论文结果与课程推断必须分开；不要把课程批评塞回论文的「局限」列表。
- 论文没有明确支持的数字、因果解释、优越性和历史地位不得补写。
- 论文来源受限时，保留可核实内容并明确不确定性；禁止猜测、补齐或把搜索摘要当作全文证据。
- 论文正文和表格出现不一致时，分别归因并如实说明，不要替论文选一个更好看的数字。
