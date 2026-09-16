---
title: OpenAI MRCR 数据集卡（一手来源）
type: source-summary
tags: [source, benchmark, long-context, mrcr, openai]
created: 2026-09-13
last_updated: 2026-09-13
source_url: https://huggingface.co/datasets/openai/mrcr
snapshot: downloads/mrcr-dataset.md
source_count: 1
confidence: medium
status: active
---

# OpenAI MRCR 数据集卡

## 摘要

本页概括 OpenAI 官方数据集卡 `openai/mrcr`（MIT 许可）2026-09-13 所见内容 [^src-mrcr-dataset]。任务：多轮合成对话（gpt-4o 生成回复，needle 与干扰项同分布）中藏有 2、4 或 8 个相同请求，模型返回第 i 个实例并前置随机字母数字 hash；hash 缺失记 0 [^src-mrcr-dataset]。判分细节：卡片文字称正确含 hash 后比较 stripped 答案，但官方示例代码只做 `startswith` 校验、`removeprefix` 去前缀、直接 `SequenceMatcher(...).ratio()`，无 whitespace strip 语句，文字与代码不一致 [^src-mrcr-dataset]。规模：438 实体、10 种写作格式；每桶 100 样本；桶按 prompt+answer 的 token 数划 8 档 [^src-mrcr-dataset]。Changelog：2025-04-12 初版；2025-12-05 bugfix，约 10% 样本 needle 过多、约 5% gold 错误，重传并加 `date_added` 字段 [^src-mrcr-dataset]。边界：卡片无「v2」命名或版本指向字段；官方代码只输出 ratio，不含 EM [^src-mrcr-dataset]。

## 快照与出处

原始快照存于 `downloads/mrcr-dataset.md`（内容为 `https://huggingface.co/datasets/openai/mrcr/raw/main/README.md` 原样）。结构导航：主条目 [[trace-as-state]]；论文侧 "MRCRv2" 命名核查与本卡代码/论文判分口径的逐条对照见 [[trace-as-state-reproduction]]。

[^src-mrcr-dataset]: [[source-mrcr-dataset]] —— OpenAI, *OpenAI MRCR: Long Context Multiple Needle in a Haystack Benchmark*，数据集卡 <https://huggingface.co/datasets/openai/mrcr>（快照：raw/main 分支 README），访问日期 2026-09-13。
