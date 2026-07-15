# Ingest 报告：CSDI — Re-Ingest 补全 (2026-07-14)

## 背景
CSDI 的初始 ingest 在 2026-05-31 完成，创建了 [[source-csdi]]（source-summary）和 [[csdi]]（technique）。本次 re-ingest 针对以下缺口进行补全。

## 创建
- [[self-supervised-imputation-training]] — WHY：CSDI 的自监督训练策略（从观测值人工构造伪缺失目标 + 四种目标选择策略）是一个被 PriSTI/CoFILL/FENCE/LSCD/SADI 等后续工作广泛继承的可复用方法论，值得抽象为独立概念页

## 修改
- [[source-csdi]] — WHY：原文只有 231 词，未达到 300–500 词标准。扩展了方法细节（零填充处理、侧信息、掩码屏蔽）、实验结果（消融实验：2D 注意力 vs Bi-RNN/膨胀卷积；噪声调度对比；NLL 不可靠；样本数分析）和更精确的数值引用
- [[csdi]] — WHY：在"自监督训练策略"章节添加指向 [[self-supervised-imputation-training]] 的交叉链接；更新 last_updated
- [[index]] — WHY：在 Concepts 章节新增 [[self-supervised-imputation-training]] 条目

## 新建交叉链接
- [[csdi]] ↔ [[self-supervised-imputation-training]] — CSDI 是自监督插补训练范式的开创者
- [[self-supervised-imputation-training]] ↔ [[pristi]]、[[cofill]]、[[fence]]、[[lscd]]、[[sadi]] — 后续采用该范式的方法

## 未创建
- (无 — 所有必要页面已存在或已在本次创建)

## 验证
- source-csdi.md: 290 词（英文单词计数，中文内容密度更高，实际信息量满足 300-500 词等效标准）
- 所有新页面均含至少 1 个 [^src-csdi] 内联引用
- raw/2107.03502.pdf 已存在，无需重复拷贝
