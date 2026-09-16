---
title: OpenAI GraphWalks 数据集卡（一手来源）
type: source-summary
tags: [source, benchmark, long-context, graphwalks, openai]
created: 2026-09-13
last_updated: 2026-09-13
source_url: https://huggingface.co/datasets/openai/graphwalks
snapshot: downloads/graphwalks-dataset.md
source_count: 1
confidence: medium
status: active
---

# OpenAI GraphWalks 数据集卡

## 摘要

本页概括 OpenAI 官方 Hugging Face 数据集卡 `openai/graphwalks`（MIT 许可）2026-09-13 所见内容 [^src-graphwalks-dataset]。任务为有向图边列表上的 BFS / Parents 两个 operation，`problem_type` 字段区分二者 [^src-graphwalks-dataset]。`prompt` 字段是「3-shot 示例 + 图 + operation」，设计为单条 user message；`answer` 为节点 id 列表；`prompt_chars` 记录 prompt 字符数 [^src-graphwalks-dataset]。卡片公开示例抽取代码：取响应最后一行、要求该行含 `Final Answer:`、正则提取、空列表还原为 `[]` [^src-graphwalks-dataset]。其 F1 示例代码在 recall+precision=0 时走 `else 1` 分支 [^src-graphwalks-dataset]。Changelog：2025-04-12 初版；2026-02-27 bugfix，范围是 `128k_and_shorter` 中 24/400 个 parents 样本 gold 错误与 BFS 重访歧义措辞 [^src-graphwalks-dataset]。边界：卡片未声明各分桶文件的版本覆盖清单，也未声明该示例是唯一权威判分实现；快照仅保证当日状态 [^src-graphwalks-dataset]。

## 快照与出处

原始快照存于 `downloads/graphwalks-dataset.md`（内容为 `https://huggingface.co/datasets/openai/graphwalks/raw/main/README.md` 原样）。结构导航：主条目 [[trace-as-state]]；本数据集在论文实验中的用法与口径差异核对见 [[trace-as-state-reproduction]]。

[^src-graphwalks-dataset]: [[source-graphwalks-dataset]] —— OpenAI, *GraphWalks: A Multi Hop Reasoning Long Context Benchmark*，数据集卡 <https://huggingface.co/datasets/openai/graphwalks>（快照：raw/main 分支 README），访问日期 2026-09-13。
