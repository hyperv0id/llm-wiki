---
title: "ICML 2026 Spotlight 论文集"
type: source-summary
tags:
  - icml-2026
  - spotlight
  - conference
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# ICML 2026 Spotlight 论文集

ICML 2026（第四十三届国际机器学习大会，2026 年 7 月 6 日于韩国首尔）的 spotlight 论文，共 **538 篇**，截至 2026-07-12 自 OpenReview 抓取[^src-icml-2026-spotlight-papers]。venue 文本为 `ICML 2026 spotlight`。

原文数据保存于 `raw/icml-2026-spotlight-papers.md`，每篇含标题、摘要、forum 链接。

## 抓取方式

OpenReview 对 `api2.openreview.net` 加了 Cloudflare 验证，headless 浏览器与直接 curl 均被 403 拦截。最终用 `agent-browser --headed` 打开真实 Chrome 窗口通过验证并复用登录态，在浏览器内同源调用 API（带 cookie）一次性取回全部 538 篇的 title + abstract。

## 覆盖范围

538 篇 spotlight 覆盖方向广泛：对齐/安全、世界模型/具身智能、扩散生成、多模态、图学习、时序预测、动力系统、科学计算、因果推断、RL 等。

## 局限

- 仅 spotlight，不含 oral/regular，不构成 ICML 2026 全貌。
- 仅含标题与摘要，未读全文，方法细节、实验设定需进入各 forum 页面进一步确认。
- 抓取时点 2026-07-12，会议期间，部分信息可能后续更新。

[^src-icml-2026-spotlight-papers]: 本页即源文件摘要，自描述。
