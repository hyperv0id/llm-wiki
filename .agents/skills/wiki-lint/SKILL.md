---
name: wiki-lint
description: "全库一致性核查：frontmatter、引用定义、source_count、悬空 wikilink、公式格式、时效性。仅在用户明确要求 lint 或核查 wiki 一致性时使用；提问与闲聊不触发。"
disable-model-invocation: true
user_invocable: true
---

# wiki-lint：全库核查

用户要求核查 `wiki/` 的一致性。schema 与文体规则在根目录 `AGENTS.md`，本技能只列检查项与报告格式。

机械项先跑脚本，再人工看剩下的语义项：

```bash
python scripts/lint.py
```

## 检查项

### 结构

- 缺 frontmatter 的页面；`type` 不在 concept / technique / entity / source-summary / analysis 内。
- `status: superseded` 但缺 `superseded_by`；`superseded_by` 指向不存在的页。
- `status: disputed` 但正文没有 `## 争议` 章节。

### 引用

- 事实性论断句没有 `[^src-*]`。
- 有 `[^src-*]` 引用但页脚无定义；定义指向的 `source-*` 页不存在。
- `source_count` 与正文唯一引用数不一致（分别报少报与多报）。
- `source_count: 1` 却含概括性陈述（「大致上」「一般来说」）。

### 公式

- 用 Unicode 数学符号拼公式（希腊字母、求和/积分号、上下标字符、`√`）。
- `$...$` 未成对闭合；`$$...$$` 未独立成行。

### 链接

- 孤立页面：其他页面里 `[[wikilink]]` 数为 0。
- 正文提及但无自身页面的概念 / 实体。
- 相关页面之间缺相互链接。

### 时效性

- `last_updated` 超过 30 天且 `status: active`。
- `confidence: high` 但 `source_count < 2`。

## 报告格式

```markdown
## Lint 报告 — YYYY-MM-DD

### 严重（必须修复）
- [ ] page.md — 具体问题

### 警告（应当修复）
- [ ] page.md — 具体问题

### 信息（最好修复）
- [ ] page.md — 具体问题
```

每条附修复建议，写到 `lint-reports/YYYY-MM-DD.md`。**先报告，等用户批准再改**；分批应用时逐批列改了什么。修完在 `wiki/log.md` 追加：

```
## [YYYY-MM-DD] lint | 全库 Lint（N 页）
{各级问题计数与已应用批次}
```

## 边界

- 不因为核查而顺手改写文案风格或断言强度；只修结构、引用、链接、格式。
- `source_count` 与脚注 key 这类全局约定，改动前单独确认。
