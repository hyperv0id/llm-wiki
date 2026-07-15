# Ingest 报告：ClimateAR — Multi-Scale Autoregressive Generative Modeling for Climate Forecasting

## 创建
- wiki/source-climatear.md — WHY：论文源摘要，记录 ClimateAR 作为首个 VAR 气候概率预测模型的核心贡献（对齐分词器、混合尺度条件控制、噪声增强 teacher-forcing）、实验性能（ACC +37.56%）、局限与引用。
- wiki/climatear.md — WHY：ClimateAR 实体页面，详述 VAR 架构细节（分段码本 VQ、跨域对齐、双层条件控制）、三阶段训练流程（VQ-VAE → AR 预训练 → AR 微调）、关键性能指标与零样本能力。
- wiki/mixed-scale-conditioning.md — WHY：混合尺度条件控制技术页面，独立阐述 intra-scale mix token 和 hybrid-scale prompt 的双层设计动机、数学形式化与消融验证，具跨领域可复用性。

## 修改
- wiki/weather-foundation-model.md — WHY：在"生成式概率预报"路线下新增 ClimateAR 条目，补充 VAR 范式作为自回归概率预测的新路线（区别于 Swift 的一致性模型路线）；添加交叉引用和脚注。
- wiki/generative-time-series-forecasting.md — WHY：新增"自回归生成式方法（VAR）"小节描述 ClimateAR，在方法对比表中添加 ClimateAR 行，补充相关页面链接和脚注。
- wiki/index.md — WHY：在 Sources、Entities、Techniques 中分别添加 source-climatear、climatear、mixed-scale-conditioning 条目。

## 新建交叉链接
- [[climatear]] ↔ [[source-climatear]]
- [[climatear]] ↔ [[mixed-scale-conditioning]]
- [[climatear]] ↔ [[weather-foundation-model]]
- [[climatear]] ↔ [[generative-time-series-forecasting]]
- [[mixed-scale-conditioning]] ↔ [[multi-scale-attention]]
- [[source-climatear]] ↔ [[source-climax]]（同为气候预测模型源摘要）
