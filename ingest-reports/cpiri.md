# Ingest 报告：CPiRi

## 创建
- wiki/source-cpiri.md — WHY：ICLR 2026 源文件，提出 CI+CD 融合的排列不变 MTSF 框架
- wiki/cpiri.md — WHY：CPiRi 是首个通过时空解耦 + 通道打乱训练系统性地解决 CD 模型位置过拟合问题的框架，需要实体页记录其设计理念、对比其他范式、汇总实验结果

## 修改
- wiki/channel-independence.md — WHY：新增 CPiRi 作为 CI+CD 折中策略的第三种范式（与 Crossformer 的全 CD、CVPE 的局部 CD 注入对比），添加通道打乱诊断的暴露性发现（Informer +400% 错误率）
- wiki/cross-dimension-dependency.md — WHY：新增 CPiRi 作为 CI-CD 融合的新范式，强调其通过时空解耦将 CI/CD 分配到不同组件的设计区别于 CVPE 的折中方案

## 新建交叉链接
- [[cpiri]] ↔ [[channel-independence]]
- [[cpiri]] ↔ [[cross-dimension-dependency]]
- [[cpiri]] ↔ [[crossformer]]
- [[cpiri]] ↔ [[itransformer]]
- [[cpiri]] ↔ [[patchtst]]
- [[cpiri]] ↔ [[mtgnn]]
- [[source-cpiri]] ↔ [[cpiri]]
