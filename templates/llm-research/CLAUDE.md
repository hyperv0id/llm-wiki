# Wiki Template — LLM Research

> 针对 LLM/ML 研究主题优化的起始模板。

## 项目背景

- **主题**: LLM / ML 研究
- **目的**: 积累论文、博客、实现笔记，构建概念图谱
- **主要语言**: zh

## 建议 wiki/ 结构

```
wiki/
  sources/        # 论文/博客摘要
  models/         # 模型 (GPT-N, Llama, Claude 等)
  techniques/     # 算法 (RLHF, DPO, LoRA 等)
  concepts/       # 思想 (scaling laws, attention 等)
  entities/       # 研究者、组织、产品
  benchmarks/     # 评估集
  analyses/       # 综合分析
```

## 额外 Frontmatter 字段 (建议)

- `paper_year: YYYY` — 仅 source-summary 类型
- `venue: "NeurIPS 2024"` 等
- `arxiv: "2401.xxxxx"` — 如有

## Citation 规则

遵循根目录 `CLAUDE.md`。论文 ingest 时如有 arxiv id，请在 source-summary 中注明。
