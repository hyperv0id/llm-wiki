---
title: "Today Queue — 多项目架构"
created: 2026-04-23
owner: yoo
---

# Today Queue — 多项目切换

优先级：从上到下依次处理。每项独立提交。

---

## ~~[MP-01] 仓库布局重新设计~~ ✅ 完成 (2026-04-23)
- 结果: `plans/architecture-multiproject.md`
- 提交: 8c0750d

---

## ~~[MP-02] `projects/` 根目录 + `projects.json` 注册表~~ ✅ 完成 (2026-04-23)
- 结果: `projects/`, `projects.json`, `templates/{CLAUDE.md,llm-research,reading-log,personal-notes}`
- 提交: 18b0cd9

---

## ~~[MP-03] `server.py` 项目 resolver (部分完成)~~ ⚠️ 仅基础 (2026-04-23)
- 完成: `dashboard/project_registry.py` 模块 + `/api/projects*` 端点 + legacy 后备
- 提交: bcf7f32
- 剩余任务(移交至 MP-07): 在现有 do_ingest/do_query/do_lint 等中将 `WIKI_DIR`/`RAW_DIR` 替换为基于 resolver 的路径

---

## [MP-04] 现有内容 → `projects/karpathy-llm/` 迁移 🚨 阻塞
- 风险度: high → 禁止自主模式 (§21.8)。需要用户批准。
- 参见 `plans/blocked.md` [BLOCK-MP-04]
- 影响范围: 根目录大规模移动
- 完成标准:
  - 使用 `git mv` 保留历史迁移
  - 在 `projects.json` 中注册 `karpathy-llm` + active
  - 将 `.dashboard-settings.json` 的 model 值复制到项目 `.settings.json`
  - 更新根目录 `README.md` 的路径示例
  - 服务器重启后确认 dashboard 正常工作

---

## ~~[MP-05] `/api/projects` CRUD 端点~~ ✅ 完成 (MP-03 中同时实现, 2026-04-23)

原项目如下 — MP-03 提交(bcf7f32) 中已完成 `/api/projects`, `/api/projects/create`,
`/switch`, `/update`, `/delete` 全部功能。

---

## [MP-05-archived] `/api/projects` CRUD 端点
- 目标: 项目列表/创建/切换/删除 API
- 影响范围: `server.py`
- 完成标准:
  - `GET /api/projects` → 列表 + active
  - `POST /api/projects` (name, description, model, template) → 创建新 `projects/<slug>/` + 起始 CLAUDE.md + 空 wiki/raw
  - `POST /api/projects/switch` (slug) → 更新 active
  - `POST /api/projects/delete` (slug, confirm) → 删除 (建议移至垃圾箱)
  - `POST /api/projects/<slug>/settings` (model) → 保存项目级模型
- 风险度: medium

---

## [MP-06] 项目模板 CLAUDE.md
- 目标: 新项目创建时提供可复制的起始模板
- 影响范围: 新增 `templates/CLAUDE.md`
- 完成标准:
  - 通用用途 (保持默认 frontmatter/citation 规则，删除领域示例)
  - 3~5 个主题变体 (llm-research / product-ops / personal-notes / reading-log) — 创建时可选择
- 风险度: low

---

## ~~[MP-07] 现有 API 端点 project 作用域~~ ✅ 完成 (2026-04-23 ~ 2026-04-24)
- Partial (读取): cb04d81 — `/api/wiki`, `/api/folders`, `/api/hash`, `/api/schema`, `/api/provenance`, `/api/index/status` 添加 `?project=<slug>`，未知 slug 返回 404
- Full (写入/Claude 调用): 1f50ddb — 所有 `do_*` + CRUD + `run_claude` cwd + GitManager + `assert_writable` 全面作用域

---

## [MP-07-archived] 现有 API 端点 project 作用域
- 目标: `/api/ingest, /api/query, /api/lint, /api/lint/fix, /api/reflect, /api/write, /api/compare, /api/review/*, /api/search, /api/page*, /api/folder, /api/slides, /api/revert, /api/history, /api/provenance, /api/suggest/sources, /api/raw/integrity, /api/index/*, /api/schema, /api/wiki, /api/folders, /api/hash, /api/query-stats, /api/assistant` — 全部接受 project 作用域
- 影响范围: 所有 handler
- 完成标准:
  - body 或 querystring 接受 `project` 字段
  - 省略时使用 active
  - 响应中 echo `project`
- 风险度: medium

---

## ~~[MP-08] 头部项目选择器 (UI)~~ ✅ 完��� (2026-04-24)
- 提交: fb39871
- 头部 `<select#projectSelect>` + 创建/删除按钮 + 2 种模态框 (New Project / Delete Project)
- 通过 `window.fetch` monkey-patch 自动为所有 `/api/*` 调用注入 `CURRENT_PROJECT` (无需修改现有 fetch 代码)
- Cmd/Ctrl+P 快捷键 → 聚焦项目选择器
- 模型选择器自动反映当前项目模型

---

## [MP-08-archived] 头部项目选择器 (UI)
- 目标: 在 dashboard 顶部添加项目下拉框 (模型选择器旁)
- 影响范围: `dashboard/index.html`
- 完成标准:
  - 项目列表 + active 显示
  - 切换时重新加载整个视图 (`/api/wiki?project=<slug>` 等)
  - "新建项目"按钮 → 模态框 (name, description, model, template 变体)
  - 模型选择器联动：读写当前项目的模型
  - 键盘快捷键: Cmd/Ctrl+P (项目切换面板)
- 风险度: medium

---

## ~~[MP-09] 项目内"按目的分类文件夹"模板支持~~ ✅ 完成 (2026-04-24)
- 提交: df51718
- `project_registry.TEMPLATE_FOLDERS` + `recommended_folders()`
- `create_project` 中根据模板自动 mkdir
- `/api/templates` 端点 + New Project 模态框中预览推荐文件夹
- 现有文件夹选择下拉框通过 `loadFolders()` 自动刷新，可直接使用

---

## [MP-09-archived] 项目���"按目的分类文件夹"模板支持
- 目标: 满足用户"按目的将页面整理到不同文件夹"的需求
- 影响范围: 页面创建流程
- 完成标准:
  - 在模板 CLAUDE.md 中明确推荐文件夹结构 (如 `sources/ entities/ concepts/ techniques/ analyses/`，或根据变体有所不同)
  - Ingest 时根据 type 自动放置到适当子文件夹的选项
  - 页面创建模态框中添加"文件夹快速选择"下拉框 (项目根目录 + 现有文件夹)
  - 侧边栏"目的"标签页: 基于 frontmatter.tags 或 folder 的分组视图
- 风险度: low

---

## ~~[MP-10] 文档更新~~ ✅ 部分完成 (2026-04-24)
- 提交: dfcf66a
- 在 README.md / README-ko.md 中添加 'Multi-project' / '多项目' 章节
- 在 Repository layout 中反映 projects/ templates/ plans/ logs/ project_registry.py
- API curl 示例 + 各模板推荐文件夹表 + legacy 兼容说明
- 剩余工作: 更新 dashboard 内 Guide 模态框，重新拍摄截图 — 待用户确认后继续

---

## [MP-10-archived] Obsidian / git / Dashboard 文档更新
- 目标: 多项目切换后用户实际可用状态 → 更新 README/指南 (CLAUDE.md §4.5)
- 影响范围: `README.md`, `README-ko.md`, `docs/`, dashboard 内 Guide 模态框
- 完成标准:
  - 反映新路径/命令
  - 重新拍摄截图/GIF (如需要)
  - 确认 `.obsidian/` vault 路径是否仍能工作 — 决定是否需要为各项目单独注册 vault
- 风险度: low