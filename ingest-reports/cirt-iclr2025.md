# Ingest 报告：CirT — Geometry-Inspired Spherical Transformer for S2S Climate Forecasting (ICLR 2025)

## 创建
- wiki/source-cirt.md — WHY：源文件摘要，CirT 论文核心贡献（圆形分块 + 傅里叶域自注意力，球面几何偏置，S2S 直接预测）
- wiki/cirt.md — WHY：CirT 模型实体页面，架构、关键指标、与其他天气模型的关系对比
- wiki/subseasonal-to-seasonal-forecasting.md — WHY：S2S 预测作为独立概念，填补"可预测性荒漠"的方法论空白
- wiki/spherical-geometry-inductive-bias.md — WHY：球面几何归纳偏置作为通用设计理念，解释平面投影的两类失真及多种解决方案
- wiki/circular-patching.md — WHY：圆形分块技术细节，分块策略对比表，与傅里叶变换的协同关系
- wiki/fourier-self-attention.md — WHY：傅里叶域自注意力技术，与 FEDformer FEB/FEA 的详细对比，DFT→注意力→IDFT 全流程

## 修改
- wiki/weather-foundation-model.md — WHY：新增"几何感知 S2S 预报"路线条目，添加 CirT 及 S2S 相关页面链接
- wiki/source-climax.md — WHY：添加 CirT 交叉引用（ClimaX 在论文中作为基线被显著超越）

## 新建交叉链接
- [[cirt]] ↔ [[source-cirt]]
- [[cirt]] ↔ [[circular-patching]]
- [[cirt]] ↔ [[fourier-self-attention]]
- [[cirt]] ↔ [[spherical-geometry-inductive-bias]]
- [[cirt]] ↔ [[subseasonal-to-seasonal-forecasting]]
- [[cirt]] ↔ [[weather-foundation-model]]
- [[cirt]] ↔ [[source-climax]]
- [[circular-patching]] ↔ [[fourier-self-attention]]
- [[circular-patching]] ↔ [[spherical-geometry-inductive-bias]]
- [[fourier-self-attention]] ↔ [[frequency-enhanced-block]]
- [[fourier-self-attention]] ↔ [[frequency-enhanced-attention]]
- [[subseasonal-to-seasonal-forecasting]] ↔ [[extreme-weather-forecasting]]
- [[spherical-geometry-inductive-bias]] ↔ [[weather-foundation-model]]
