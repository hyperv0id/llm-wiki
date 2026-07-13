# Ingest 报告：DistDF (ICLR 2026)

## 创建
- **wiki/source-distdf.md** — WHY: DistDF 论文的 source-summary 页面，覆盖核心贡献（自相关偏差定理、联合分布 Wasserstein discrepancy、Bures-Wasserstein 实现）、实验设置与结果、局限性
- **wiki/autocorrelation-bias.md** — WHY: 论文核心概念之一，MSE 在标签自相关时的偏差，独立于论文而具有一般性意义
- **wiki/joint-distribution-wasserstein-alignment.md** — WHY: DistDF 的核心技术——用联合分布 Wasserstein discrepancy 对齐条件分布，模型无关的学习目标范式

## 修改
- **wiki/index.md** — WHY: 添加 source-distdf（Sources）、autocorrelation-bias（Concepts）、joint-distribution-wasserstein-alignment（Techniques）条目并更新 last_updated
- **wiki/log.md** — WHY: 记录 ingest 操作

## 新建交叉链接
- [[source-distdf]] ↔ [[autocorrelation-bias]] — DistDF 提出 autocorrelation bias 理论
- [[source-distdf]] ↔ [[joint-distribution-wasserstein-alignment]] — DistDF 的核心技术实现
- [[autocorrelation-bias]] ↔ [[joint-distribution-wasserstein-alignment]] — 问题与解决方案的关系