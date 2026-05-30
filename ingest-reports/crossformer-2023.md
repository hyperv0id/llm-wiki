# Ingest 报告：Crossformer (Zhang & Yan, ICLR 2023)

## 创建
- wiki/source-crossformer-2023.md — WHY：源文件摘要页，对应 raw/crossformer-2023.pdf
- wiki/crossformer.md — WHY：首个显式利用跨维度依赖的 MTS Transformer，实体页
- wiki/cross-dimension-dependency.md — WHY：核心概念——MTS 中变量间关联，CI vs CD 争论的关键
- wiki/dsw-embedding.md — WHY：Crossformer 的核心嵌入方法，2D 向量阵列保留维度信息
- wiki/two-stage-attention.md — WHY：Crossformer 的核心注意力机制，分两阶段处理时间和维度
- wiki/router-mechanism-for-cross-dimension.md — WHY：降低跨维度注意力复杂度的路由机制，后被 CVPE 借鉴
- wiki/hierarchical-encoder-decoder-ts.md — WHY：Crossformer 的多尺度架构，对长预测提升显著

## 修改
- wiki/channel-independence.md — WHY：添加 Crossformer 作为全 CD 架构对比项，补充 CI vs CD 讨论
- wiki/patch-based-tokenization.md — WHY：添加 DSW embedding 作为 2D 分段嵌入对比项
- wiki/lstf.md — WHY：添加 Crossformer 到 LSTF 模型列表，扩展发展趋势
- wiki/cvpe.md — WHY：修正 Crossformer 链接为 wikilink 格式
- wiki/index.md — WHY：添加所有新页面到索引
- wiki/log.md — WHY：记录 ingest 操作

## 新建交叉链接
- [[crossformer]] ↔ [[cross-dimension-dependency]]
- [[crossformer]] ↔ [[dsw-embedding]]
- [[crossformer]] ↔ [[two-stage-attention]]
- [[crossformer]] ↔ [[router-mechanism-for-cross-dimension]]
- [[crossformer]] ↔ [[hierarchical-encoder-decoder-ts]]
- [[crossformer]] ↔ [[cvpe]] (Router 机制借鉴关系)
- [[crossformer]] ↔ [[channel-independence]] (CI vs CD 对比)
- [[crossformer]] ↔ [[lstf]] (LSTF 模型列表)
- [[dsw-embedding]] ↔ [[patch-based-tokenization]] (分段嵌入对比)
- [[router-mechanism-for-cross-dimension]] ↔ [[router-attention-for-cvpe]] (借鉴关系)
- [[router-mechanism-for-cross-dimension]] ↔ [[adaptive-graph-agent-attention]] (agent token 降低复杂度类比)
