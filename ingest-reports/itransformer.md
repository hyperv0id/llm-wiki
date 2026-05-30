# Ingest 报告：iTransformer (ICLR 2024)

## 创建
- wiki/sources/source-itransformer.md — WHY：核心源文件摘要，详细记录反转架构动机、组件职责重定义、框架通用性、变量泛化、消融实验和局限
- wiki/entities/itransformer.md — WHY：实体页面，完整架构流程、与其他模型的对比表格、性能亮点和局限分析
- wiki/concepts/multivariate-correlation-attention.md — WHY：新概念——iTransformer 的核心创新，attention 作用于变量维度捕获多变量相关性，具有可解释性
- wiki/techniques/variate-token-embedding.md — WHY：新技术——将整条变量序列嵌入为独立 token，与 temporal token 和 patch token 的对比
- wiki/techniques/inverted-transformer-architecture.md — WHY：新技术——反转组件维度而不修改组件的架构范式，含框架通用性和消融验证

## 修改
- wiki/channel-independence.md — WHY：新增 iTransformer 作为 CI 与 CD 的第三条路径（独立嵌入 + attention 关联），添加交叉引用和源文件引用
- wiki/crossformer.md — WHY：新增与 iTransformer 的详细对比表格（token 化/跨变量交互/时间依赖/组件修改/性能），添加引用
- wiki/cross-dimension-dependency.md — WHY：新增 iTransformer 反转范式作为建模跨维度依赖的新方式，添加引用
- wiki/patch-based-tokenization.md — WHY：新增 variate token 作为 patch token 极端情况的对比，添加引用
- wiki/lstf.md — WHY：新增 iTransformer 在 LSTF 时间线中的位置，记录其解决回看窗口增长性能不提升的突破
- wiki/index.md — WHY：添加新页面到 Sources/Entities/Concepts/Techniques 各分类
- wiki/log.md — WHY：记录 ingest 操作

## 新建交叉链接
- [[itransformer]] ↔ [[channel-independence]]
- [[itransformer]] ↔ [[crossformer]]
- [[itransformer]] ↔ [[cross-dimension-dependency]]
- [[itransformer]] ↔ [[patch-based-tokenization]]
- [[itransformer]] ↔ [[lstf]]
- [[itransformer]] ↔ [[informer]]
- [[variate-token-embedding]] ↔ [[patch-based-tokenization]]
- [[variate-token-embedding]] ↔ [[channel-independence]]
- [[multivariate-correlation-attention]] ↔ [[cross-dimension-dependency]]
- [[multivariate-correlation-attention]] ↔ [[crossformer]]
- [[inverted-transformer-architecture]] ↔ [[channel-independence]]
