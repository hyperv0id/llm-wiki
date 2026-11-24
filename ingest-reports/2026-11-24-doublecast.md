# Ingest 报告：doublecast (CAF-7M & DoubleCast)

- **分级**：Tier 1（同时交付经经验 CRPS 效用验证的大规模多模态语料 CAF-7M 与基于冻结 Chronos 解码器的 DualT5 条件化基线 DoubleCast，在数据侧正面回应「文本融合为何常败给单模态」的核心缺口）

## 创建
- wiki/doublecast.md — WHY：构建 CAF-7M + DoubleCast 的 Tier 1 独立机制与基准页面，涵盖人话大义、CRPS 效用判据、分阶段 DualT5 解码块 `aligned` 维度流、五向跨文对照及三行定性实验判词
- wiki/source-doublecast.md — WHY：记录 ICML 2026 论文（Zheng*, Marcotte* et al.）核心摘要、去黑话配方、三行实验判词与缺口

## 修改
- wiki/multimodal-time-series-forecasting.md — WHY：在基准对照表补入 CAF-7M 行，新增「容量约束 vs 数据效用验证」二维正交设计矩阵，补充关联索引与引用（`source_count` 22→23）
- wiki/constrained-text-fusion.md — WHY：在谱系位置表补充与 CAF-7M/DoubleCast 的失效归因对立与跨槽位正交对照（`source_count` 5→6）
- wiki/time-mmd.md — WHY：在谱系表补充「防泄漏时间戳对齐 → 经验 CRPS 效用验证」瓶颈接力 Δ 并压缩九域表行数以过质量门禁（`source_count` 6→7）
- wiki/chronos.md — WHY：补充 DoubleCast 对冻结 Chronos（T5-large）解码器的 DualT5 文本条件化扩展及 GIFT-Eval / MSPG 复现对照（`source_count` 9→10）
- wiki/aurora.md — WHY：补充同生态位原生生成式预训练与冻结骨干解码器适配的路线与干扰文本退化对照，并清理表内禁用符号（`source_count` 5→6）
- wiki/circular-dependency-in-multimodal-fusion.md — WHY：补充 CAF-7M 测试集采用 GPT-5.2 既作 CRPS 效用筛选器又作评测对象的同构自我验证偏置案例（`source_count` 1→2）
- wiki/index.md — WHY：在 Sources 段按字母序插入 `[[source-doublecast]]`，在 Entities 段按字母序插入 `[[doublecast]]`
- wiki/log.md — WHY：追加本次 Tier 1 入库结论性日志

## 新建交叉链接（标明关系类型）
- [[doublecast]] ↔ [[constrained-text-fusion]]（同失效现象归因对立与跨槽位正交：数据侧 CRPS 效用筛选 vs 融合层低秩容量约束）
- [[doublecast]] ↔ [[time-mmd]]（瓶颈接力 Δ：防泄漏时间戳对齐 → 经验预测效用验证筛选）
- [[doublecast]] ↔ [[chronos]]（条件化扩展与复现对齐：冻结 T5-large 编码器 + 解码块 DualT5 文本交叉注意力，GIFT-Eval 保留原始数值精度）
- [[doublecast]] ↔ [[aurora]]（同生态位并发对照：原生多模态 Flow Matching 生成式预训练 vs 冻结骨干判别式/自回归适配器，及乱序/错配文本退化对照）
- [[doublecast]] ↔ [[circular-dependency-in-multimodal-fusion]]（同构风险：测试集效用筛选器与被测模型重合引入自我验证偏置）
