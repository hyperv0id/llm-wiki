---
title: "Architecture — 多项目仓库布局"
created: 2026-04-23
status: draft
owner: yoo
scope: MP-01
---

# Multi-Project Architecture (MP-01)

## 1. 目标

将当前固定在根目录的单一 wiki (`wiki/`, `raw/`, `CLAUDE.md`) 隔离到 `projects/<slug>/` 子目录下，
以实现在一个 dashboard 中有选择地运营多个独立主题的 wiki。

原则:

- **隔离(Isolation)**: 一个项目的 ingest/query/lint 不会读取或写入其他项目的数据。
- **单一进程**: 服务器只有一个。项目切换是状态变更(层切换)，而非进程重启。
- **向下兼容(Legacy mode)**: 迁移前服务器也能正常工作。如果没有 `projects.json`，则将当前根目录的 `wiki/raw/` 视为 "default" 项目。
- **git 连续性**: 文件移动使用 `git mv`。保留历史。

---

## 2. 目录树 (目标状态)

```
.
├── projects/
│   ├── karpathy-llm/                    ← 迁移后的当前 wiki
│   │   ├── CLAUDE.md                    (项目级 schema)
│   │   ├── .settings.json               (项目级 model 等)
│   │   ├── raw/                         (immutable)
│   │   │   └── assets/
│   │   ├── wiki/
│   │   │   ├── index.md
│   │   │   ├── log.md
│   │   │   ├── overview.md
│   │   │   ├── sources/                 (MP-09 推荐文件夹)
│   │   │   ├── entities/
│   │   │   ├── concepts/
│   │   │   ├── techniques/
│   │   │   └── analyses/
│   │   ├── ingest-reports/
│   │   ├── reflect-reports/
│   │   ├── plans/
│   │   │   ├── today-queue.md
│   │   │   ├── backlog.md
│   │   │   └── blocked.md
│   │   └── query-log.jsonl
│   └── <future-project>/
│       └── ... (相同结构)
├── projects.json                        ← 项目注册表 (保留在根目录)
├── templates/
│   └── CLAUDE.md                        ← 起始模板 (MP-06 填充)
├── dashboard/                           ← UI 服务器 (不依赖项目)
├── logs/                                ← 自主模式会话日志 (保留在根目录)
├── plans/                               ← 多项目切换用计划 (切换完成后移至 projects/karpathy-llm/plans/)
├── CLAUDE.md                            ← 根目录 schema (仅保留通用规则)
├── README.md / README-ko.md             ← 不依赖项目的文档
└── .obsidian/                           ← 根目录 vault 设置
```

### 保留在根目录 vs 下沉到项目作用域

| 项目 | 位置 | 依据 |
|------|------|------|
| `dashboard/` | 根目录 | 服务器单一。切换项目。 |
| `projects.json` | 根目录 | 注册表文件 (项目列表, active) |
| `templates/` | 根目录 | 新项目创建时的复制源 |
| `logs/` | 根目录 | 自主模式会话日志是服务器级工作记录 |
| `.obsidian/` | 根目录 | 在一个 Obsidian vault 中浏览所有项目 (Q-2 待定) |
| `.gitignore` | 根目录 | repo 级 |
| `README.md` | 根目录 | 不依赖项目的介绍 |
| `CLAUDE.md` | 根目录 | 通用规则 (有此文件则项目 CLAUDE.md 优先级较低) |
| `wiki/`, `raw/`, `ingest-reports/`, `reflect-reports/`, `query-log.jsonl` | 项目 | 内容 |
| `.dashboard-settings.json` | **删除** → 分散到 `projects/<slug>/.settings.json` | |
| 项目级 `plans/` | 项目 | 项目级工作队列 |

### 迁移前 (legacy mode) 行为

- 如果 `projects.json` 文件不存在，服务器进入 "legacy" 状态。
- `get_project(name=None)` → 如果没有 name，返回 legacy 路径 (当前的 `wiki/ raw/ CLAUDE.md`)。
- Dashboard 头部显示 "项目: (legacy)" + 迁移引导按钮。

---

## 3. `projects.json` schema

```json
{
  "version": 1,
  "active": "karpathy-llm",
  "projects": [
    {
      "slug": "karpathy-llm",
      "title": "Karpathy LLM Wiki",
      "description": "Andrej Karpathy 相关 LLM 资料收集",
      "model": "claude-opus-4-7",
      "created": "2026-04-22",
      "last_used": "2026-04-23",
      "template": "llm-research"
    }
  ]
}
```

字段规则:
- `slug` — `make_slug()` 结果。字母数字 + 连字符 + unicode(中文) 允许。禁止重复。目录名。
- `active` — 当前选中的项目 slug。没有则 legacy。
- `model` — `AVAILABLE_MODELS` 的 id 之一。与项目 `.settings.json` 同步。
- `template` — 新建时复制的模板变体名称 (MP-06)。

---

## 4. Resolver API (MP-03 中实现)

### 4.1 核心函数

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Project:
    slug: str                 # "karpathy-llm" | "" (legacy)
    is_legacy: bool           # projects.json 不存在时为 True
    root: Path                # projects/<slug>/ or PROJECT_ROOT
    wiki_dir: Path            # <root>/wiki
    raw_dir: Path             # <root>/raw
    claude_md: Path           # <root>/CLAUDE.md
    settings_file: Path       # <root>/.settings.json (legacy 时为 PROJECT_ROOT/.dashboard-settings.json)
    ingest_reports: Path
    reflect_reports: Path
    plans_dir: Path
    query_log: Path
    title: str
    model: str                # 当前模型
```

```python
def list_projects() -> list[dict]: ...          # projects.json 解析
def get_active_slug() -> str | None: ...
def get_project(slug: str | None = None) -> Project: ...
def create_project(slug: str, title: str, description: str, model: str, template: str) -> Project: ...
def switch_project(slug: str) -> Project: ...
def delete_project(slug: str, confirm: bool) -> dict: ...
def update_project_settings(slug: str, **fields) -> Project: ...
```

### 4.2 端点作用域规则

- 所有现有端点在 body/query 中接受 `project` 字段。
- 省略时使用 `get_active_slug()`。如果没有 active 且是 legacy，则使用 legacy 路径。
- 响应中 echo `project: <slug>`。

### 4.3 `run_claude(...)` 签名变更

- `cwd=str(PROJECT_ROOT)` → 改为 `cwd=str(project.root)`。
- 模型参数从 `project.model` 解析 (不再是全局 SETTINGS)。
- 读取 `CLAUDE.md` 时使用 `project.claude_md` 路径。

### 4.4 git 策略

- **保持单一 repo (选项 A, Q-1 默认值)**。按项目子目录提交。
- `GitManager._stage_all()` → `add projects/<slug>/wiki/ projects/<slug>/raw/ projects/<slug>/ingest-reports/`
- 提交消息前缀包含项目 slug: `ingest(karpathy-llm): <title>`
- 分支策略不变 — 所有工作在 feature 分支进行。

### 4.5 安全措施

- `assert_writable(path)` — raw/ 不可变性**适用于所有项目的 raw/** (不仅是当前项目)。
- `assert_raw_create_only(path)` — 同样扩展。
- 拒绝 resolver 中跨越项目边界的路径访问 (`../`)。

---

## 5. 迁移步骤 (MP-04 中执行，禁止自主模式)

1. `git mv wiki projects/karpathy-llm/wiki`
2. `git mv raw projects/karpathy-llm/raw`
3. `git mv ingest-reports projects/karpathy-llm/ingest-reports`
4. `git mv reflect-reports projects/karpathy-llm/reflect-reports`
5. `git mv query-log.jsonl projects/karpathy-llm/query-log.jsonl`
6. `git mv CLAUDE.md projects/karpathy-llm/CLAUDE.md` — 在根目录写新的精简版 `CLAUDE.md` (通用规则 + 项目规则请参考 projects/<slug>/CLAUDE.md)
7. 将 `.dashboard-settings.json` 转换为 `projects/karpathy-llm/.settings.json` 后删除原文件
8. 创建新的 `projects.json` (active: "karpathy-llm")
9. 更新 `README.md` 路径示例
10. 单一冒烟测试 (服务器重启 → `/api/projects` → `/api/wiki?project=karpathy-llm`)

---

## 6. 待决事项 (参见 `plans/blocked.md`)

- Q-1 单一 repo vs 每项目独立 repo → 本文假设 **选项 A (单一 repo)**
- Q-2 Obsidian vault 作用域 → 保持当前根目录 vault，项目级分离作为单独任务。
- Q-3 slug 规则 → 复用 `make_slug()` + 重复检查。
- Q-4 删除策略 → 移动到 `projects/.trash/<slug>-<ts>/` (hard delete 选项需要 `confirm=hard` 必需)。

---

## 7. 实现步骤检查清单 (由此设计派生)

- [ ] MP-02: 创建 `projects/` 目录 + `projects.json` 初始文件 + `templates/CLAUDE.md` 存根
- [ ] MP-03: resolver + legacy 后备 + 添加至少一个端点(`/api/projects`) + 进程内冒烟测试
- [ ] MP-04 (阻塞): 移动现有内容 — 需要用户批准
- [ ] MP-05~MP-10: 参见上级计划