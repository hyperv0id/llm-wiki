# Ingest 报告：UrbanFM (arXiv 2026)

## 创建
- `wiki/source-urbanfm.md` — WHY：UrbanFM 是一篇 2026 年 arXiv 预印本，以 scaling 为第一性原理系统性地推进城市时空基础模型，必须创建 source-summary 页面记录其三大贡献（WorldST/MiniST/极简架构）和 EvalST 基准实验结果
- `wiki/urbanfm.md` — WHY：UrbanFM 是一个独立的城市时空基础模型实体，与 UrbanDiT、UniFlow、UniST、OpenCity、UrbanGPT 等形成对比分工，需要创建 entity 页面记录架构设计、性能数据和与其他模型的对比表

## 修改
- `wiki/spatio-temporal-foundation-model.md` — WHY：UrbanFM 是时空基础模型领域的重要新工作，以 scaling 为中心视角明显区别于 UrbanDiT（扩散）、UniFlow（memory retrieval）、UrbanGPT（LLM）等现有路线，添加到 Single-Modal 模型列表中
- `wiki/opencity.md` — WHY：OpenCity 的 Related Pages 已涵盖 UniST/UniFlow/UrbanDiT 等，补加 UrbanFM 作为同领域的最新 scaling 视角基础模型
- `wiki/unist.md` — WHY：UniST 的 Related Pages 覆盖了 UrbanDiT（同 lab 后续工作），UrbanFM 作为不同研究组但同领域的 scaling 视角工作值得交叉引用
- `wiki/uniflow.md` — WHY：UniFlow 的 Connection 页面已涵盖 UrbanDiT/UrbanGPT/OpenCity 等，UrbanFM 的 WorldST 数据规模远超 UniFlow 所用数据集，构成有意义的对比视角
- `wiki/urbangpt.md` — WHY：UrbanGPT 的 Related Pages 已涵盖 OpenCity/UniFlow/UrbanDiT 等，UrbanFM 的极简 Transformer 架构与 UrbanGPT 的 LLM 路线形成鲜明对比
- `wiki/traffic-forecasting.md` — WHY：traffic-forecasting 页面在 foundation model 段落中已详细描述 UrbanGPT/MoST/OpenCity，UrbanFM 作为零样本性能最强的时空基础模型（MAPE 优于现有 39-70.2%），必须加入以保持该段落的完整性
- `wiki/index.md` — WHY：新增 source-urbanfm 和 urbanfm 两个页面，按约定更新索引
- `wiki/log.md` — WHY：按 CLAUDE.md 规范记录 ingest 操作

## 新建交叉链接
- [[urbanfm]] ↔ [[spatio-temporal-foundation-model]]
- [[urbanfm]] ↔ [[opencity]]
- [[urbanfm]] ↔ [[unist]]
- [[urbanfm]] ↔ [[uniflow]]
- [[urbanfm]] ↔ [[urbangpt]]
- [[urbanfm]] ↔ [[traffic-forecasting]]
