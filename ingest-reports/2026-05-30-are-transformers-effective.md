# Ingest 报告：Are Transformers Effective for Time Series Forecasting?

## 创建
- wiki/source-zeng-2022-are-transformers-effective.md — WHY：源文件摘要页面，记录 Zeng et al. (2022) 的核心论点（自注意力排列不变性导致时序信息丢失）和 LTSF-Linear 基线的详细实验发现
- wiki/ltsf-linear.md — WHY：该论文提出的 LTSF-Linear（DLinear/NLinear/Vanilla Linear）是 LTSF 领域的关键基线实体，对现有 Transformer 模型提出根本性质疑

## 修改
- wiki/informer.md — WHY：添加"Critical Reassessment"章节和 Zeng et al. 引用，记录逐步简化 Informer 到线性模型反而提升性能的发现
- wiki/autoformer.md — WHY：添加 LTSF-Linear 交叉引用和 Zeng et al. 引用，记录 Autoformer 在打乱输入顺序时性能下降 56.91% 的实验
- wiki/fedformer.md — WHY：添加 LTSF-Linear 交叉引用和 Zeng et al. 引用，记录 FEDformer 因频域归纳偏置而在 ETTh1 上最具竞争力的发现
- wiki/lstf.md — WHY：添加"Linear Model Challenge"章节和 Zeng et al. 引用，记录 LTSF-Linear 对 Transformer LSTF 范式的根本性质疑
- wiki/instance-normalization.md — WHY：添加 LTSF-Linear/NLinear 交叉引用和 Zeng et al. 引用，记录 NLinear 的减法归一化是 RevIN 的前身思想
- wiki/index.md — WHY：添加新页面条目
- wiki/log.md — WHY：记录 ingest 操作

## 新建交叉链接
- [[ltsf-linear]] ↔ [[informer]]
- [[ltsf-linear]] ↔ [[autoformer]]
- [[ltsf-linear]] ↔ [[fedformer]]
- [[ltsf-linear]] ↔ [[lstf]]
- [[ltsf-linear]] ↔ [[instance-normalization]]
- [[source-zeng-2022-are-transformers-effective]] → [[ltsf-linear]]
