# Lint 报告 — 2026-07-07

检查范围：2026-07-07 摄入的 17 个 `source-summary` 页面。
检查日期：2026-07-08（当前工作目录）；页面 `created` 字段：2026-07-07。

---

## 严重（必须修复）

### 1. source-exost.md — source_count 与实际引用数不匹配

- **文件：** `wiki/source-exost.md`
- **问题：** Frontmatter `source_count: 0`（第 11 行），但正文中包含 6 处 `[^src-exost]` 引用（1 个唯一源文件）。
- **修复：** 将 `source_count: 0` 改为 `source_count: 1`。
- **引用统计：** 正文引用 `[^src-exost]` 出现 6 次，全部指向同一源文件。

---

## 警告（应当修复）

### 2-11. 10 个页面 confidence: high 但 source_count < 2

根据 CLAUDE.md 字段规则：`confidence: high` 要求"多个源文件一致支持"。单一源文件的 source-summary 页面应使用 `confidence: medium`（"1~2 个源文件，无反驳"）。

| # | 页面 | 字段位置 | 当前值 | 建议值 |
|---|------|---------|--------|--------|
| 2 | `source-cast.md` | 第 13-14 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 3 | `source-climax.md` | 第 13-14 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 4 | `source-gpt4mts.md` | 第 12-13 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 5 | `source-exollm.md` | 第 11-12 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 6 | `source-from-news-to-forecast.md` | 第 11-12 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 7 | `source-stllm.md` | 第 10-11 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 8 | `source-terra.md` | 第 12-13 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 9 | `source-timexer.md` | 第 11-12 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 10 | `source-stg-mamba.md` | 第 12-13 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |
| 11 | `source-raf.md` | 第 12-13 行 | `confidence: high`, `source_count: 1` | `confidence: medium` |

### 12. source-multimodal-pinn.md — 指向不存在的页面的 wiki 链接

- **文件：** `wiki/source-multimodal-pinn.md`
- **问题：** 第 55 行 `[[physics-informed-neural-network]]` 指向一个在 `wiki/` 中不存在的页面（无 `physics-informed-neural-network.md` 或 `concepts/physics-informed-neural-network.md` 等文件）。
- **修复选项：**
  - a) 创建 `wiki/physics-informed-neural-network.md` 页面
  - b) 如果该概念已由其他页面覆盖，改为指向已有页面的链接
  - c) 移除该链接，改为纯文本

---

## 信息（最好修复）

### 13-18. 缺少「交叉链接」或「相关链接」章节

以下 6 个页面完全缺少 `## 交叉链接` / `## 相关链接` 章节，未建立与其他 wiki 页面的双向链接图谱：

| # | 页面 | 说明 |
|---|------|------|
| 13 | `source-cast.md` | 无交叉链接章节 |
| 14 | `source-climax.md` | 无交叉链接章节 |
| 15 | `source-gpt4mts.md` | 无交叉链接章节 |
| 16 | `source-exost.md` | 无交叉链接章节 |
| 17 | `source-pi-mfm.md` | 无交叉链接章节 |
| 18 | `source-exotst.md` | 无交叉链接章节 |

### 19. source-exost.md — 正文中提及但未使用 wikilink 的既有页面

- **文件：** `wiki/source-exost.md`
- **问题：** 正文第 56 行和 60 行提及以下已有对应 wiki 页面的概念/模型，但未使用 `[[wikilink]]`：
  - `GWNet` → `source-gwnet.md` 存在
  - `STGCN` → `source-stgcn.md` 存在
  - `DCRNN` → `source-dcrnn.md` 存在
  - `TimeXer` → `source-timexer.md` 存在（该行同属基准对比列表）
- **建议：** 将以上模型名称替换为 `[[source-gwnet|GWNet]]`、`[[source-stgcn|STGCN]]`、`[[source-dcrnn|DCRNN]]` 等 wikilink 形式，丰富图谱连接。

---

## 检查摘要

| 类别 | 数量 |
|------|------|
| 已检查的 source-summary 页面 | 17 |
| 严重问题 | 1 |
| 警告问题 | 11（10 个 confidence 问题 + 1 个不存在的页面引用） |
| 信息类问题 | 7（6 个缺失交叉链接章节 + 1 个遗漏 wikilink） |
| 无问题的页面 | `source-stfm-pipeline-review`, `source-whatif-tsf`, `source-solar-vlm` |

### 关于 source-select-then-balance（已取代）

`source-select-then-balance.md`（在 2026-07-07 更新）已被正确标记：
- `status: superseded` ✓
- `superseded_by: [[source-exost]]` ✓
- 含有 `> [!warning] 已取代` callout ✓
- 通过检查，无需修复。