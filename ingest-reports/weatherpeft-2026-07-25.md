# Ingest 报告：WeatherPEFT (ICLR 2026)

## 创建
- wiki/source-weatherpeft.md — WHY：源文件摘要，WeatherPEFT 论文的 source-summary 页面，300-500 字覆盖核心论题、TADP+SFAS 方法、三任务实验与贡献/局限
- wiki/weatherpeft.md — WHY：WeatherPEFT 实体页面，首个 WFM 参数高效微调框架，需要独立 entity 页面链接到 TADP/SFAS 技术页并反向引用 source-summary
- wiki/task-adaptive-dynamic-prompting.md — WHY：TADP 是 WeatherPEFT 的核心前向技术组件，有独立的双阶段（内部模式提取 + 外部模式整合）流程，值得作为 technique 页面归档以便与其他 prompt tuning 方法交叉对比
- wiki/stochastic-fisher-guided-adaptive-selection.md — WHY：SFAS 是 WeatherPEFT 的核心反向技术组件，Fisher 信息 + 退火随机性 + Top-k 选择的设计足够独到，需作为 technique 页面与 Child-Tuning/SCT/EWC 等方法形成交叉引用网络

## 修改
- wiki/weather-foundation-model.md — WHY：概念页面需新增"高效微调"章节和 WeatherPEFT 条目，因 PEFT 是 WFM 实用部署的关键瓶颈，WeatherPEFT（ICLR 2026）是此方向的里程碑工作；source_count 3→4，last_updated 更新
- wiki/index.md — WHY：按 Ingest 工作流更新内容目录，在 Sources/Entities/Techniques 分别添加新页面条目

## 新建交叉链接
- [[weatherpeft]] ↔ [[weather-foundation-model]]
- [[weatherpeft]] ↔ [[task-adaptive-dynamic-prompting]]
- [[weatherpeft]] ↔ [[stochastic-fisher-guided-adaptive-selection]]
- [[task-adaptive-dynamic-prompting]] ↔ [[weather-prompt]]（TADP vs WeatherGFM visual prompt 对比）
- [[stochastic-fisher-guided-adaptive-selection]] ↔ [[projected-fisher-divergence]]（Fisher 信息的不同用途）
- [[weather-foundation-model]] → [[weatherpeft]]（新反向链接）
