---
title: "Backlog — 多项目外的额外改进候选"
created: 2026-04-23
---

# Backlog — 额外实现/开发需求项目

与多项目切换无关，基于对当前实现的反思而识别的改进候选。
优先级由用户决定。

---

## 运营 / 稳定性

- **[OPS-01] 任务队列 + 进度流式传输** — `/api/ingest` 时间长有 HTTP 超时风险。使用 SSE/WS 推送进度日志 + 后台任务标识符。
- **[OPS-02] 长期运行 Claude 调用日志持久化** — 当前仅 stdout。失败时难以追踪原因。保存 `runs/<date>-<id>.log`。
- **[OPS-03] 速率限制 / 预算保护** — 按模型累计 token/费用 + 超过阈值时阻止。当前仅在 `query-log.jsonl` 中记录费用。
- **[OPS-04] 备份/恢复** — 项目级 zip 导出/导入。也可考虑 git bundle。
- **[OPS-05] 健康检查改进** — 在一个端点中检查 Obsidian vault 打开状态、git 状态、Claude CLI 响应时间。

## 质量 / 功能

- **[FEAT-01] 跨项目搜索** — 同时搜索多个项目 wiki。当前 TF-IDF 仅适用于单个 wiki。
- **[FEAT-02] 项目间链接/嵌入** — 允许引用其他项目页面 — 类似 `[[projectA::page]]` 的语法。
- **[FEAT-03] 基于标签的浏览器** — 按 frontmatter `tags` 过滤/分组。当前 UI 主要基于 type。
- **[FEAT-04] 源文件上传 UX** — 当前仅支持 title/content 文本输入。支持文件上传(.pdf, .html, .md)。
- **[FEAT-05] 页面历史视图** — 逐页 git blame/diff 查看器 (在 dashboard 中阅读)。
- **[FEAT-06] 自动 reflect 调度** — 定期执行 reflect → 堆积到建议队列。
- **[FEAT-07] Diff 预览** — ingest/lint-fix 应用前确认 diff 后确认。
- **[FEAT-08] 多语言 wiki 管道** — 连接同一概念的 KO/EN 页面 (translation 关系，非 superseded)。

## 架构 / 治理

- **[GOV-01] 自动矛盾检测** — LLM 检查新 claim 是否与现有 claim 冲突，并向用户发出警告。
- **[GOV-02] Citation 验证器 (本地)** — 无需 Claude 调用，使用 regex + frontmatter 自动执行 lint。CI hook 候选。
- **[GOV-03] Source trust score** — 各 source 的可信度字段 (peer-reviewed / blog / tweet 等) + 自动计算页面 confidence。
- **[GOV-04] CHANGELOG** — 项目级 CHANGELOG.md (Keep a Changelog 格式)。ingest/reflect/lint 自动追加。

## 安全 / 访问控制

- **[SEC-01] 确认禁止非 localhost 访问** — 当前绑定 `::` — 文档补充 + 选项化，确保仅暴露本地。
- **[SEC-02] 项目删除保护** — `confirm` 参数必需 + 经过垃圾箱(trash/)。
- **[SEC-03] 密钥扫描** — ingest 时警告 raw/wiki 中是否包含 API 密钥/令牌模式。

## 测试 / DX

- **[DX-01] 单元测试** — 对 `server.py` 的 `make_slug`, `parse_fm`, `_diff_snapshots`, `_tokenize` 等纯函数进行 pytest。
- **[DX-02] 端点契约测试** — 对各 `/api/*` 进行冒烟测试。
- **[DX-03] 开发模式热重载** — 当前需手动重启。
- **[DX-04] 日志格式标准化** — JSON 行日志 + 级别。

## 用户体验

- **[UX-01] 入门向导** — 首次运行时"创建项目" → "添加第一个 source" → "尝试提问" 3 步教程。
- **[UX-02] 命令面板** — Cmd/Ctrl+K → fuzzy search 所有功能 + 项目切换。
- **[UX-03] 移动端布局** — 当前仅桌面端。根据 §8 至少支持最低程度。
- **[UX-04] 深色/浅色主题切换** — 当前固定深色。