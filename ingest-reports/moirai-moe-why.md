# Ingest 报告：Moirai-MoE (ICML 2025)

## 创建
- wiki/source-moirai-moe.md — WHY：300-500 词 source-summary 覆盖 Moirai-MoE 核心贡献（MoE token 级专业化、簇基门控、next-token prediction、39 数据集实验、三项模型分析发现），符合 CKLAUDE.md 规范
- wiki/moirai-moe.md — WHY：首个 MoE 时间序列基础模型的实体页，与 Moirai 对比表、模型配置、关键性能数字、Time-MoE 关系说明
- wiki/token-level-specialization.md — WHY：概念页阐述数据驱动 token 级专业化范式，与频率级/数据集级专业化对比，含工作机制和优势分析
- wiki/cluster-based-gating.md — WHY：技术页详述簇基门控机制，从预训练 dense 模型提取聚类中心引导 MoE 路由，含与标准线性门控的消融对比

## 修改
- wiki/mixture-of-experts.md — WHY：新增"时间序列基础模型中的应用"小节，以 Moirai-MoE 作为首个将 Sparse MoE 用于 TSFM 预训练的案例，source_count 5→6
- wiki/timesfm.md — WHY：新增 Moirai-MoE 零样本对比条目（Moirai-MoE-B 在 CRPS 总评上超越 TimesFM，MMoE-S 在部分数据集上优于但总体不及），source_count 6→7
- wiki/chronos.md — WHY：新增 Moirai-MoE 对比条目（65× 更少激活参数、273s vs 551s 推理速度），source_count 5→6

## 新建交叉链接
- [[moirai-moe]] ↔ [[mixture-of-experts]]
- [[moirai-moe]] ↔ [[token-level-specialization]]
- [[moirai-moe]] ↔ [[cluster-based-gating]]
- [[mixture-of-experts]] ↔ [[token-level-specialization]]
- [[mixture-of-experts]] ↔ [[cluster-based-gating]]
- [[timesfm]] ↔ [[moirai-moe]]
- [[chronos]] ↔ [[moirai-moe]]
- [[token-level-specialization]] ↔ [[cluster-based-gating]]

## Raw 文件
raw/moirai-moe-empowering-time-series-foundation-models-with-sparse-mixture-of-experts-icml2025.pdf
