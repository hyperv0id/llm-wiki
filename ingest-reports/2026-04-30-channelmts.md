# Ingest 报告：ChannelMTS

## 创建

- **wiki/source-channelmts.md** — WHY：ChannelMTS KDD 2026 论文的详细摘要，包含背景、框架设计、实验结果、与 UniCA 对比、部署经验等
- **wiki/channelmts.md** — WHY：ChannelMTS 实体页面，提供核心问题、技术架构、实验结果概览
- **wiki/retrieval-augmented-statistical-channel.md** — WHY：RAGC 技术专题页，详细解释检索增强统计信道的算法流程、与传统 RAG 的区别

## 修改

- **wiki/multimodal-time-series-forecasting.md** — WHY：添加 ChannelMTS 与 UniCA 的对比分析，更新 source_count 从 1 到 2
- **wiki/index.md** — WHY：添加 source-channelmts、channelmts、retrieval-augmented-statistical-channel 三个新页面到相应类别
- **wiki/log.md** — WHY：添加 2026-04-30 ingest 记录

## 交叉链接

| 页面 | 新增链接 |
|------|----------|
| [[channelmts]] | → [[source-channelmts]], → [[multimodal-time-series-forecasting]] |
| [[source-channelmts]] | → 无（source-summary 本身） |
| [[retrieval-augmented-statistical-channel]] | → [[channelmts]], → [[source-channelmts]] |
| [[multimodal-time-series-forecasting]] | → [[channelmts]], → [[source-channelmts]] |

## 源文件

- **raw/channelmts-kdd-2026.pdf** — 从 Zotero storage HR6IDTTA 复制

## 笔记

ChannelMTS 与 UniCA 的关系：
- 都处理多模态时间序列预测
- UniCA 针对 TSFMs 的异构协变量适应（分类/图像/文本）
- ChannelMTS 专门针对高铁信道预测，使用环境快照（位置+K因子+延迟）
- 核心区别：UniCA 用协变量同质化，ChannelMTS 用环境-信道对齐（median+IQR）+ RAGC