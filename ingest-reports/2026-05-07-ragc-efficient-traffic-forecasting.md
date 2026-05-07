# Ingest 报告：RAGC — Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution

## 创建
- wiki/source-ragc-efficient-traffic-forecasting.md — WHY：源文件摘要页面，记录论文核心论点、三项贡献、实验结果和局限
- wiki/ragc.md — WHY：模型实体页面，完整记录 RAGC 的架构、技术创新、实验结果和消融分析
- wiki/efficient-cosine-operator.md — WHY：ECO 是论文最核心的技术创新，独立为技术页面便于交叉引用和与其他线性复杂度方法对比
- wiki/stochastic-shared-embedding.md — WHY：SSE 原为推荐系统技术，RAGC 首次引入交通预测，独立页面便于追踪跨领域技术迁移
- wiki/residual-difference-mechanism.md — WHY：RDM 是使 SSE 可用于时空预测的关键机制，独立页面便于引用其理论分析
- wiki/node-embedding-regularization.md — WHY：节点嵌入过参数化是自适应图学习的普遍问题，独立为概念页面便于未来其他论文引用

## 修改
- wiki/traffic-forecasting.md — WHY：添加 "Regularized Adaptive Graph Convolution" 方法小节和 RAGC 引用
- wiki/large-scale-spatial-temporal-graph.md — WHY：添加 ECO 作为新的线性化方法类别，补充余弦相似度与随机特征映射的对比
- wiki/index.md — WHY：注册新页面到索引

## 新建交叉链接
- [[ragc]] ↔ [[efficient-cosine-operator]]
- [[ragc]] ↔ [[stochastic-shared-embedding]]
- [[ragc]] ↔ [[residual-difference-mechanism]]
- [[ragc]] ↔ [[node-embedding-regularization]]
- [[ragc]] ↔ [[traffic-forecasting]]
- [[ragc]] ↔ [[large-scale-spatial-temporal-graph]]
- [[efficient-cosine-operator]] ↔ [[adaptive-graph-agent-attention]] (同为空间复杂度优化方法)
- [[efficient-cosine-operator]] ↔ [[large-scale-spatial-temporal-graph]]
- [[stochastic-shared-embedding]] ↔ [[residual-difference-mechanism]]
- [[node-embedding-regularization]] ↔ [[stochastic-shared-embedding]]
