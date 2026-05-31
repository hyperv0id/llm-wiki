# Ingest 报告：UniST (2024)

## 创建
- wiki/source-unist.md — **WHY**：UniST 是 KDD 2024 上首个 one-for-all 时空基础模型，建立非 LLM 路线的通用城市时空预测范式，20+ 数据集、6.71M 参数、零样本超越少样本基线，要求创建标准 source-summary 页面
- wiki/unist.md — **WHY**：UniST 的核心架构创新（时空 patching、四种互补掩码策略、双记忆池 prompt 学习）构成可复用的方法模板，且与 UrbanDiT（同实验室后续工作）构成明确演进链，需独立 technique 页面记录

## 修改
- wiki/index.md — **WHY**：将新 source-summary 和 technique 页面添加到索引的 Sources 和 Techniques 分类
- wiki/log.md — **WHY**：记录完整的 ingest 操作、创建/更新页面清单、核心创新点摘要
- wiki/spatio-temporal-foundation-model.md — **WHY**：已有 UniST 纯文本提及但缺少 wikilink，添加 `[[unist]]` 链接和更完整的描述
- wiki/opencity.md — **WHY**：两处 UniST 纯文本提及（对比分析第 25 行、性能对比第 92 行）转换为 `[[unist|UniST]]` wikilink
- wiki/traffic-forecasting.md — **WHY**：Foundation Model 子节中已有 UrbanGPT/MoST/OpenCity 条目但缺少 UniST，补充为时空基础模型路线的奠基工作
- wiki/gpt-st.md — **WHY**：GPT-ST 和 UniST 共享 MAE 预训练范式，GPT-ST 为 per-dataset、UniST 为 cross-dataset one-for-all，在 Historical Context 中添加演进关系说明
- wiki/urbandit.md — **WHY**：UrbanDiT 和 UniST 同属清华 FIB Lab（Yuan Yuan 为两篇的共同作者），在相关页面中添加 `[[unist|UniST]]` 链接标记承接关系
- wiki/urbangpt.md — **WHY**：在 Comparison with Other ST Foundation Models 表格中 UrbanGPT/UrbanDiT/GPT-ST/OpenCity 均已列出，补充 UniST 列展示完整的非 LLM vs LLM 路径对比

## 新建交叉链接
- [[unist]] ↔ [[spatio-temporal-foundation-model]]
- [[unist]] ↔ [[opencity]]
- [[unist]] ↔ [[urbangpt]]
- [[unist]] ↔ [[urbandit]]
- [[unist]] ↔ [[gpt-st]]
- [[unist]] ↔ [[traffic-forecasting]]
- [[unist]] ↔ [[mae]]
- [[unist]] ↔ [[std-mae]]
- [[unist]] ↔ [[patchtst]]
- [[source-unist]] ↔ [[unist]]
