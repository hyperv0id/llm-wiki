# Ingest 报告：DistDF (ICLR 2026) + 2026-07-17 gap-fill

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

## 2026-07-17 gap-fill 补全

### 新创建
- **wiki/bures-wasserstein.md** — WHY: BW 度量是 DistDF 的实用实现（Lemma 3.5），也被 Gaussian SB 使用，值得独立页面以集中描述该数学工具

### 补全的交叉链接
- [[joint-distribution-wasserstein-alignment]] ↔ [[optimal-transport]] — WHY: Wasserstein 距离是 OT 理论的产物
- [[joint-distribution-wasserstein-alignment]] ↔ [[direct-forecast]] — WHY: DistDF = Distribution-aware Direct Forecast
- [[joint-distribution-wasserstein-alignment]] ↔ [[bures-wasserstein]] — WHY: DistDF 的核心度量实现
- [[source-distdf]] → [[optimal-transport]], [[direct-forecast]] — WHY: 论文的 OT 理论基础与 DF 范式归属
- [[optimal-transport]] → [[joint-distribution-wasserstein-alignment]], [[bures-wasserstein]] — WHY: OT 在时序预测中的 Wasserstein 对齐应用
- [[direct-forecast]] → [[joint-distribution-wasserstein-alignment]] — WHY: DistDF 是 DF 的分布感知扩展

### 修改的页面
- [[joint-distribution-wasserstein-alignment]] — 新增 3 个 Related Techniques 链接 + frontmatter 更新
- [[source-distdf]] — 新增 Related Paradigms 节 + Key Terminology 增加 BW 链接
- [[optimal-transport]] — 新增 2 个 相关页面 反向链接
- [[direct-forecast]] — Related 新增 joint-distribution-wasserstein-alignment 链接
- [[index]] — Techniques 新增 bures-wasserstein 条目