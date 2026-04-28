# Wiki Template (generic)

> 此文件是创建新项目时复制到 `projects/<slug>/CLAUDE.md` 的起始模板。
> 请根据项目的主题/领域自由修改。根目录 `CLAUDE.md` 的通用规则始终保留。

## 项目背景

- **主题**: {{TOPIC}}
- **目的**: {{PURPOSE}}
- **主要语言**: zh

## 目录结构

```
raw/              # 不可变原始源文件 (禁止修改/删除)
wiki/             # LLM 维护的页面
  sources/        # source-summary 类型
  entities/       # 专有名词
  concepts/       # 思想/框架
  techniques/     # 方法论/算法
  analyses/       # 综合分析
  index.md
  log.md
  overview.md
ingest-reports/
reflect-reports/
plans/
```

`wiki/` 子文件夹为建议项。是否严格应用由项目管理员决定。

## Frontmatter 规则

完全遵循根目录 `CLAUDE.md` 的 Frontmatter 规则。如需项目特定额外字段，请在此文件中说明。

## Ingest 工作流

遵循根目录 `CLAUDE.md` 的 "Ingest 工作流"。

## Lint 检查清单

遵循根目录 `CLAUDE.md` 的 Lint 检查清单。

## 项目特定风格指南

(如有额外规则，请在此处编写)
