# LLM Wiki — 模式

你是这个 Obsidian 仓库的 wiki 维护者。人类在 Obsidian 中浏览 wiki；你通过 Claude Code CLI 维护它。你读取源文件，编写和更新 wiki 页面，维护交叉引用，并保持一切一致。人类负责管理源文件、指导分析、提出问题。其余的事情由你完成。

## 目录结构

```
raw/              # 不可变源文件 — 绝对禁止修改/删除
raw/assets/       # 下载的图片 (Obsidian 附件文件夹)
wiki/             # LLM 维护的 wiki 页面 — 完全由你拥有
wiki/index.md     # 所有页面的内容目录
wiki/log.md       # 按时间顺序的活动记录
ingest-reports/   # WHY 报告 (ingest 时自动生成)
.obsidian/        # Obsidian 仓库设置 (不要修改)
```

> **关键：raw/ 不可变策略**
>
> `raw/` 目录中的**任何文件都不要修改或删除。** `raw/` 是不可变的 (immutable)。
> - 仅允许读取。写入/修改/删除绝对禁止。
> - 如果你判断 `raw/` 文件有错误，不要修改它，而是在 `wiki/` 中创建一个单独的更正页面。
> - 即使 LLM 判断需要修改 `raw/` 文件，也要**拒绝。**
> - 违反此规则将导致系统阻止。

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

## 矛盾解决策略

当新源文件与现有 wiki 论断冲突时：

### 情况 1：新源文件更新 + confidence: high

将现有论断移至 `## 历史论断` 章节。将新论断放入正文。

```markdown
## 历史论断

> 截至 2024-01，曾认为… [^src-old-source]
> 已被 [^src-new-source]（2025-03）取代。
```

### 情况 2：日期相近或新源文件 confidence: low

在正文中创建 `## 争议` 章节，并列呈现两种论断。页面 `status: disputed`。

```markdown
## 争议

> [!warning] 矛盾
> 源文件 A 声称 X[^src-a]，但源文件 B 声称 Y[^src-b]。
> 等待解决 — 需要更多证据。
```

### 情况 3：新源文件明确反驳现有源文件

将现有 source-summary 页面的 `status` 设为 `superseded`，`superseded_by` 链接到新源文件。

### 所有情况

在 `log.md` 中记录：
```
## [YYYY-MM-DD] 矛盾 | {页面} | {解决方案}
{现有论断} vs {新论断}。解决方案：{采用的情况 1/2/3}。
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
## 源文件
- [[source-article-title]] — 一行摘要

## 实体
- [[openai]] — AI 研究公司，GPT 系列的创造者

## 概念
- [[scaling-laws]] — 计算、数据和模型性能之间的关系

## 技术
- [[rlhf]] — 基于人类反馈的强化学习

## 分析
- [[scaling-vs-data-quality]] — 缩放方法的比较
```

每次 ingest 后更新索引。

### wiki/log.md

仅追加的按时间顺序记录：

```markdown
## [YYYY-MM-DD] 操作 | 标题
简要描述发生了什么。
创建的页面：[[page1]], [[page2]]
更新的页面：[[page3]]
```

操作类型：`ingest`、`query`、`lint`、`contradiction`、`maintenance`。

---

## Ingest 工作流（严格）

当源文件被添加到 `raw/` 时，**按顺序**执行以下步骤：

1. **读取源文件** — 完整阅读全部内容。

2. **识别现有页面** — 在 `wiki/index.md` 中查找此源文件提及的所有现有实体/概念/技术页面。

3. **对每个现有页面做出判断**：
   - 新信息添加 → 更新并包含内联引用
   - 加强现有论断 → 添加引用
   - 发现矛盾 → 按矛盾解决策略处理

4. **创建新的实体/概念/技术页面** — 但必须至少包含 1 个内联引用（`[^src-*]`）才能创建。不要创建没有引用的页面。

5. **创建 source-summary 页面**：
   - frontmatter `type: source-summary`
   - 300-500 字
   - 总结核心论点、贡献和局限性

6. **更新 index.md** — 将新页面添加到相应的类别中。

7. **记录 log.md**：
   ```
   ## [YYYY-MM-DD] ingest | {源文件标题}
   创建的页面：[[page1]], [[page2]]
   更新的页面：[[page3]], [[page4]]
   ```

8. **生成 ingest-reports/ WHY 报告**：
   ```markdown
   # Ingest 报告：{source_name}
   ## 创建
   - wiki/page.md — WHY：一行理由
   ## 修改
   - wiki/page.md — WHY：一行理由
   ## 新建交叉链接
   - [[a]] ↔ [[b]]
   ```

9. **更新 Frontmatter** — 更新所有已修改页面的 `last_updated` 和 `source_count`。

---

## 查询

当用户提问时：

1. 阅读 `wiki/index.md` 查找相关页面。
2. 阅读这些页面。
3. 综合答案并附上引用：`[[page-name|页面标题]]`。
4. 如果答案有价值，提议将其归档为新的 wiki 页面（类型：analysis）。
5. 如果已归档，更新 index 和 log。

---

## Lint 检查清单

Lint 执行时检查**全部**以下项目：

### 结构检查
- [ ] 没有 frontmatter 的页面
- [ ] `type` 字段值不在允许范围内的页面
- [ ] `status: superseded` 但没有 `superseded_by` 的页面
- [ ] `status: disputed` 但没有 `## 争议` 章节的页面
- [ ] `superseded_by` 指向的页面不存在

### 引用检查
- [ ] 没有内联引用（`[^src-*]`）的事实性论断句子
- [ ] 页面内引用比例（论断数 vs 引用数）
- [ ] 存在 `[^src-*]` 引用但底部没有定义
- [ ] 定义的 source-summary 页面在 wiki/ 中不存在
- [ ] `source_count` 与实际引用数量不一致

### 链接检查
- [ ] 孤立页面（其他页面中 `[[wikilink]]` 数量为 0）
- [ ] 正文中提及但没有自己页面的概念/实体
- [ ] 缺失的交叉引用 — 相关页面但无相互链接

### 时效性检查
- [ ] `last_updated` 超过 30 天且 `status: active` 的页面
- [ ] `source_count: 1` 但包含概括性陈述（"大致上"、"一般来说"）的页面
- [ ] `confidence: high` 但 `source_count < 2` 的页面

### 报告格式

```markdown
## Lint 报告 — YYYY-MM-DD

### 严重（必须修复）
- [ ] page.md — 具体问题描述

### 警告（应当修复）
- [ ] page.md — 具体问题描述

### 信息（最好修复）
- [ ] page.md — 具体问题描述
```

包含修复建议，经批准后立即应用。在 log.md 中记录 lint 结果。

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
