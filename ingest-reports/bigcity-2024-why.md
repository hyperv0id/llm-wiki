# Ingest 报告：BIGCity (arXiv 2024)

## 创建
- wiki/source-bigcity.md — WHY：创建源文件摘要页，记录 BIGCity 论文的核心贡献、ST-unit 统一表示、ST Tokenizer、VMTP 架构、两阶段训练策略、8 任务覆盖、3 城市数据集、跨城市泛化能力及局限性
- wiki/bigcity.md — WHY：创建深入的方法/技术实体页面，详细记录 BIGCity 的三级孤岛分类（STSD→MTSD→MTMD）、ST-unit 数学模型（含轨迹与交通状态的双向统一公式）、ST Tokenizer 四模块架构推导、VMTP 的 task-oriented prompt 四种模板、三头通用输出层、两阶段训练损失函数、消融实验排序、跨模态多任务训练的正向迁移效应、与其他 ST foundation model 的对比表、认识论意义分析
- ingest-reports/bigcity-2024-why.md — WHY：记录本次 ingest 的所有创建/修改/交叉链接决策

## 修改
- wiki/spatio-temporal-foundation-model.md — WHY：在 Related Pages 中添加 [[bigcity]] 链接，标注其为首个 MTMD 时空模型
- wiki/traffic-forecasting.md — WHY：在 Foundation Model 子节内新增 BIGCity 段落（含 `[^src-bigcity]` 引用 + 脚注定义），标注其为首个 MTMD 模型；source_count 从 22 → 23
- wiki/urbangpt.md — WHY：在 Related Pages 中添加 BIGCity 链接，作为 LLM-based ST 路线的 MTMD 扩展
- wiki/unist.md — WHY：在 Related Pages 中添加 BIGCity 链接，作为超越 UniST traffic-only 的 MTMD 补充
- wiki/uniflow.md — WHY：在 Connection 节中添加 BIGCity 链接，作为 traffic-only → trajectory+traffic 的范式扩展
- wiki/opencity.md — WHY：在 Related Pages 中添加 BIGCity 链接，作为 OpenCity 零样本路线的 MTMD 扩展
- wiki/urbandit.md — WHY：在 Related Resources 中添加 BIGCity 链接，作为扩散路线外的 LLM+prompt MTMD 替代方案
- wiki/gpt-st.md — WHY：在 Related Pages 中添加 BIGCity 链接，作为从 per-dataset MAE pre-training 到 universal MTMD 的路标
- wiki/index.md — WHY：添加 source-bigcity 源文件条目和 bigcity 技术页面条目
- wiki/log.md — WHY：追加 ingest 操作日志，记录创建/更新页面列表

## 新建交叉链接
- [[bigcity]] ↔ [[spatio-temporal-foundation-model]]
- [[bigcity]] ↔ [[traffic-forecasting]]
- [[bigcity]] ↔ [[urbangpt]]
- [[bigcity]] ↔ [[unist]]
- [[bigcity]] ↔ [[uniflow]]
- [[bigcity]] ↔ [[opencity]]
- [[bigcity]] ↔ [[urbandit]]
- [[bigcity]] ↔ [[gpt-st]]
