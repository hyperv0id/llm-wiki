# Ingest 报告：SparseTSF

## 创建
- wiki/source-sparsetsf.md — WHY：TPAMI 2026 论文（ICML 2024 Oral），提出 Cross-Period Sparse Forecasting 技术，首次将 LTSF 模型参数量降至 1k 以下
- wiki/sparsetsf.md — WHY：SparseTSF 模型实体页面，含核心创新、架构变体、性能对比
- wiki/cross-period-sparse-forecasting.md — WHY：Cross-Period Sparse Forecasting 技术专项说明，类属 technique 页面

## 修改
- wiki/periodicity-modeling-in-time-series.md — WHY：新增路线五"轻量级稀疏周期建模"，添加 SparseTSF 章节，更新周期性认知层级（新增第 5 级：显式—稀疏），更新 source_count (14→15)，更新时间线表格

## 新建交叉链接
- [[source-sparsetsf]] ↔ [[sparsetsf]]
- [[source-sparsetsf]] ↔ [[cross-period-sparse-forecasting]]
- [[sparsetsf]] ↔ [[cross-period-sparse-forecasting]]

## 建议后续补充
- 可创建轻量级模型对比专题页面，汇总 DLinear (~1k) → FITS (~10k) → SparseTSF (<1k) 的演进路线