# Ingest Report: craft-neurips2025.pdf

## 创建
- wiki/source-craft.md — WHY：CRAFT (NeurIPS 2025) 源文件摘要，首个跨城市零样本交通流生成模型，DDPM+RAG 混合架构
- wiki/craft.md — WHY：CRAFT 实体页面，核心创新（GFA+RCA）、架构、实验结果、消融分析
- wiki/geographic-feature-alignment.md — WHY：GFA 技术页面，双损失（TFA 流量对齐 + CCA 最优传输跨城市对齐）的域偏移解决方案
- wiki/retrieval-based-condition-augmentation.md — WHY：RCA 技术页面，检索增强条件构建（时间嵌入 + 地理表示相似度检索 + 自注意力聚合）
- wiki/cross-city-traffic-flow-generation.md — WHY：跨城市流量生成概念页面，问题域定义、挑战、与预测/基础模型的区别

## 修改
- wiki/traffic-forecasting.md — WHY：添加"Cross-City Traffic Flow Generation"章节，链接 CRAFT，扩展 traffic-forecasting 从预测到生成的覆盖
- wiki/spatio-temporal-foundation-model.md — WHY：添加 CRAFT 条目在"Traffic Flow Generation (Cross-City)"子节，定位为与 ST 基础模型互补的生成范式
- wiki/diffusion-models.md — WHY：在应用领域中添加跨城市交通流生成条目，链接 CRAFT 和 GFA/RCA
- wiki/index.md — WHY：添加 source-craft（源文件列表）、craft（实体列表）、cross-city-traffic-flow-generation（概念列表）、geographic-feature-alignment 和 retrieval-based-condition-augmentation（技术列表）
- wiki/log.md — WHY：记录本次 ingest 操作

## 新建交叉链接
- [[craft]] ↔ [[geographic-feature-alignment]]
- [[craft]] ↔ [[retrieval-based-condition-augmentation]]
- [[craft]] ↔ [[cross-city-traffic-flow-generation]]
- [[craft]] ↔ [[rast]] (RAG-for-STF)
- [[geographic-feature-alignment]] ↔ [[optimal-transport]]
- [[retrieval-based-condition-augmentation]] ↔ [[retrieval-augmented-spatio-temporal-forecasting]]
- [[cross-city-traffic-flow-generation]] ↔ [[traffic-forecasting]]
- [[cross-city-traffic-flow-generation]] ↔ [[spatio-temporal-foundation-model]]
