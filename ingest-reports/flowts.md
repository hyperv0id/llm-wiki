# Ingest 报告：FlowTS (arXiv 2025)

## 创建
- wiki/source-flowts.md — WHY：源文件摘要，FlowTS 将 rectified flow 用于时间序列生成的首次应用
- wiki/flowts.md — WHY：FlowTS 实体页面，ODE 时间序列生成模型，30 步采样 SOTA
- wiki/rectified-flow-for-time-series.md — WHY：Rectified Flow 在时间序列生成中的应用范式概念页，区别于图像域应用
- wiki/adaptive-sampling-flow-matching.md — WHY：FlowTS 的创新自适应采样策略，探索-利用权衡驱动的时间步缩放

## 修改
- wiki/flow-matching.md — WHY：添加 FlowTS、rectified-flow-for-time-series、adaptive-sampling-flow-matching 三个相关页面链接，新增 [^src-flowts] 引用
- wiki/generative-time-series-forecasting.md — WHY：新增 FlowTS 条目到流匹配方法列表，添加方法对比表格行，source_count 更新为 9
- wiki/tsflow.md — WHY：添加 FlowTS 和 rectified-flow-for-time-series 交叉引用，同为 TS 生成模型
- wiki/rectified-flow.md — WHY：新增 FlowTS 应用案例章节，展示 Rectified Flow 在时间序列域的首次应用，source_count 更新为 2
- wiki/index.md — WHY：在 Sources/Entities/Concepts/Techniques 四个分类中添加新页面条目
- wiki/log.md — WHY：记录 ingest 活动

## 新建交叉链接
- [[flowts]] ↔ [[rectified-flow]] — FlowTS 是 rectified flow 在时间序列域的实现
- [[flowts]] ↔ [[flow-matching]] — rectified flow 是 flow matching 的特例
- [[flowts]] ↔ [[tsflow]] — 同为流匹配时间序列生成模型，范式对比：rectified flow vs CFM+GP
- [[flowts]] ↔ [[dits]] — DiTS 也使用 rectified flow 用于 TS 预测
- [[flowts]] ↔ [[generative-time-series-forecasting]] — 生成式 TS 预测的新方法
- [[rectified-flow-for-time-series]] ↔ [[rectified-flow]] — 时间序列域的应用 vs 图像域基础
- [[rectified-flow-for-time-series]] ↔ [[diffusion-models]] — 扩散模型在 TS 中的替代方案
- [[adaptive-sampling-flow-matching]] ↔ [[exploration-vs-exploitation]] — 自适应采样的理论来源
- [[flowts]] ↔ [[adaptive-sampling-flow-matching]] — 模型实体 ↔ 核心技术
- [[flowts]] ↔ [[rectified-flow-for-time-series]] — 模型 ↔ 范式概念
