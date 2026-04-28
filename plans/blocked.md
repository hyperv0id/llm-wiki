---
title: "Blocked — 需要决策/信息的项目"
created: 2026-04-23
---

# Blocked

---

## [2026-04-23 20:45] BLOCK-MP-04 现有内容迁移

**原队列项目**: MP-04 将 wiki/raw/ingest-reports/reflect-reports/query-log.jsonl/CLAUDE.md 移动到 `projects/karpathy-llm/`
**尝试次数**: 0 (未在自主模式下执行)
**阻塞原因**:
- 风险度 `high` — `today-queue.md` 中明确。根据 §21.8 禁止自主模式。
- 大量 `git mv`，根目录布局变更，需要重启运行中的 dashboard 服务器(pid 94329) 并基于路径重启。
- 需先决定 Q-1~Q-4 (见下文)

**需要决策/信息** (用户):
- Q-1 单一 repo vs 每个项目独立 repo
- Q-2 Obsidian vault 作用域
- Q-3 slug 规则 (默认: `make_slug` + 重复检查)
- Q-4 删除策略 (默认: `projects/.trash/` 软删除)

**执行检查清单** (用户批准后):
1. `git mv` 5 项 (wiki, raw, ingest-reports, reflect-reports, query-log.jsonl)
2. `git mv CLAUDE.md projects/karpathy-llm/CLAUDE.md` + 在根目录重新创建精简版 CLAUDE.md
3. `.dashboard-settings.json` → `projects/karpathy-llm/.settings.json` (迁移 model 值后删除原文件)
4. 在 `projects.json` 中注册 `karpathy-llm` + 设置 `active`
5. 服务器重启后执行 `/api/projects`, `/api/wiki?project=karpathy-llm` 冒烟测试 — 注意: 如果 MP-07 未完成，`/api/wiki` 仍会引用 legacy 路径导致出错。需先完成 MP-07。

**相关提交**: 8c0750d (MP-01), 18b0cd9 (MP-02), bcf7f32 (MP-03)

---

## 等待决策的设计要点

## 等待决策的设计要点 (MP-04 执行前必须)

- **[Q-1] 是否按项目分离 git repo**
  - 选项 A: 单一 repo + `projects/<slug>/` 子目录提交 (推荐，简单)
  - 选项 B: 每个项目独立 repo
  - 决策者: 用户
  - 影响: MP-04, MP-05, OPS-04

- **[Q-2] Obsidian vault 作用域**
  - 选项 A: 整个根目录作为一个 vault (当前) — 项目间移动自由
  - 选项 B: 每个项目独立 vault 注册 — 在 obsidian.json 中添加 N 个条目
  - 决策者: 用户 (根据工作习惯)
  - 影响: MP-10

- **[Q-3] 项目 slug 规则**
  - 强制字母数字+连字符? 允许韩文? 不允许空格?
  - 备选: 用户输入 title + 自动生成 slug (复用 make_slug)

- **[Q-4] 项目删除策略**
  - 立即永久删除 vs 移动到 trash/<slug>-<timestamp>/
  - 默认推荐 trash (可恢复)