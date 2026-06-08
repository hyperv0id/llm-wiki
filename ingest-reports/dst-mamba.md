# Ingest 报告：DST-Mamba — Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction (AAAI 2025)

## 创建
- **wiki/source-dst-mamba.md** — WHY：DST-Mamba 是首个将时序分解与 Mamba SSM 结合并扩展至时空维度的交通预测框架，核心贡献在于分解策略 + 空间视角的双向 Mamba 编码
- **wiki/dst-mamba.md** — WHY：作为独立的实体页面，与 S-Mamba 形成 Mamba-based 时序预测方法的完整谱系（通用 MTSF → 时空交通预测）
- **wiki/spatio-temporal-decomposition.md** — WHY：DST-Mamba 首次将纯时序分解（Autoformer/FEDformer）扩展至时空维度，趋势用线性、季节用 Mamba 的分工是基于数据特性的合理设计取舍
- **wiki/multi-scale-linear-prediction.md** — WHY：多尺度下采样 + 自上而下混合是 DST-Mamba 趋势组件的核心技术，与 TimeMixer 的 PDM 形成对比

## 修改
- **wiki/s-mamba.md** — WHY：添加 DST-Mamba 作为相关页面（S-Mamba 的时空扩展），以及更新 last_updated
- **wiki/mamba.md** — WHY：添加 DST-Mamba 作为 Mamba 在时空交通预测领域的代表性应用

## 新建交叉链接
- [[s-mamba]] ↔ [[dst-mamba]] — Mamba-based 时序预测谱系
- [[mamba]] ↔ [[dst-mamba]] — Mamba 在时空预测中的应用
- [[spatio-temporal-decomposition]] ↔ [[autoformer]] / [[fedformer]] / [[timemixer]] — 分解方法的演进链
- [[multi-scale-linear-prediction]] ↔ [[timemixer]] / [[ltsf-linear]] — 多尺度线性预测的技术对比
