# Ingest 报告：CoRA (Covariate-Aware Adaptation of TSFMs)

## 创建
- [[source-cora]] — 源文件摘要，CoRA 框架的三大组件和实验结果
- [[cora-tsfm]] — CoRA 实体页面，详细阐述架构、设计原则和性能

## 修改
- 无

## 新建交叉链接
- [[cora-tsfm]] ↔ [[timesfm]] — CoRA 兼容 TimesFM backbone
- [[cora-tsfm]] ↔ [[chronos]] — CoRA 兼容 Chronos backbone
- [[cora-tsfm]] ↔ [[channel-independence]] — CoRA 在多元预测中采用 CI 策略
- [[cora-tsfm]] ↔ [[cross-dimension-dependency]] — CI 隐式处理的跨变量依赖概念
- [[cora-tsfm]] ↔ [[mixture-of-experts]] — Moirai-MoE 等 TSFM 的相关架构
- [[cora-tsfm]] ↔ [[unified-covariate-adaptation]] — UniCA 为直接对比方法

---

## 2026-07-04 全方位整合更新

### 新创建
- [[tsfm-covariate-adaptation-comparison]] — WHY：将 CoRA 从孤立论文置于 TSFM 协变量适配方法全景图中，系统比较六种路线
- [[zero-initialized-adaptation]] — WHY：提取 CoRA/LoRA/DiT 共有的零初始化设计原则为独立概念页

### 更新的页面
- [[cora-tsfm]] — WHY：新增 6+ wikilinks（sundial/dits/dit/flow-matching-forecasting/multimodal-ts-forecasting/tsfm-comparison/zero-init），source_count 1→4，confidence medium→high
- [[source-cora]] — WHY：添加 source-unica 引证，关系表加入 wikilinks
- [[unica]] — WHY：添加 CoRA 竞争对比（前置注入劣势+零初始化缺失），新增 limits entry
- [[dits]] — WHY：添加 CoRA 作为替代路线的位置说明
- [[mm-dit-for-time-series]] — WHY：Related Concepts 添加 CoRA 和 tsfm-comparison
- [[sundial]] — WHY：添加 CoRA 作为 Sundial 上的适配框架
- [[heterogeneous-covariates]] — WHY：添加 CoRA 作为不经同质化的替代处理路线
- [[covariate-homogenization]] — WHY：添加 CoRA 替代方案说明
- [[covariate-fusion-module]] — WHY：对比表加入 CoRA adaLN injection 行
- [[conditional-attention-pooling]] — WHY：添加 Causality Embedding 作为替代机制
- [[index]] — WHY：新增 2 页面入索引
- [[log]] — WHY：记录本次全方位整合操作

### 新建交叉链接（30+）
- [[cora-tsfm]] ↔ [[sundial]] / [[dits]] / [[dit]] / [[zero-initialized-adaptation]] / [[tsfm-covariate-adaptation-comparison]] / [[heterogeneous-covariates]] / [[multimodal-time-series-forecasting]]
- [[unica]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]（反向，此前单向）
- [[dits]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]
- [[sundial]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]
- [[heterogeneous-covariates]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]
- [[covariate-homogenization]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]
- [[covariate-fusion-module]] → CoRA 对比行
- [[conditional-attention-pooling]] → [[cora-tsfm]] / [[tsfm-covariate-adaptation-comparison]]
