# Ingest 报告：RiverMamba (NeurIPS 2025)

## 源文件
- raw/rivermamba-state-space-model-for-global-river-discharge-and-flood-forecasting.pdf
- 作者：Mohamad Hakam Shams Eddin, Yikui Zhang, Stefan Kollet, Juergen Gall (University of Bonn & Research Centre Jülich)
- 发表：NeurIPS 2025

## 创建
- wiki/source-rivermamba.md — WHY：RiverMamba 是首个全球 0.05° 河流流量/洪水预报深度学习模型，Mamba SSM + 空间填充曲线 + LOAN + Hindcast-Forecast 分层架构，在 GloFAS 再分析和 GRDC 实测站点均 SOTA，对本仓库的 Mamba/SSM/时空预测/水文应用均有重要交叉引用价值
- wiki/rivermamba.md — WHY：核心实体页，详细记录 RiverMamba 架构（输入嵌入→Hindcast→Forecast→回归头）、训练策略（洪水重现期加权损失）、关键性能指标
- wiki/space-filling-curves.md — WHY：RiverMamba 的核心技术创新——将 Peano/Hilbert 空间填充曲线引入深度学习时空建模，实现 Mamba 在大规模空间点上的线性复杂度扫描，是序列化/反序列化范式在水文领域的首创应用
- wiki/location-aware-adaptive-normalization.md — WHY：LOAN 层通过静态地理属性（集水区形态）注入位置感知偏置，是 RiverMamba 条件化归一化的关键组件，也为其他地理空间深度学习提供可复用技术
- wiki/flood-forecasting.md — WHY：洪水预报是 RiverMamba 的应用领域，涵盖从物理模型（GloFAS）到深度学习（LSTM→GNN→RiverMamba）的方法演进、洪水重现期度量、以及本仓库首次覆盖的水文 AI 主题

## 修改
- wiki/mamba.md — WHY：在 Mamba 实体页的关联页面中添加 RiverMamba（全球洪水预报方向的新 Mamba 应用），source_count 2→3
- wiki/deep-state-space-model.md — WHY：补充"在时空建模中的扩展"章节，说明 RiverMamba 如何将深度 SSM 从单一序列扩展到数万个空间点的全球尺度时空交互建模，source_count 1→2

## 交叉链接新增
- [[rivermamba]] ↔ [[mamba]]
- [[rivermamba]] ↔ [[deep-state-space-model]]
- [[rivermamba]] ↔ [[space-filling-curves]]
- [[rivermamba]] ↔ [[location-aware-adaptive-normalization]]
- [[rivermamba]] ↔ [[flood-forecasting]]
- [[flood-forecasting]] ↔ [[precipitation-nowcasting]]
- [[flood-forecasting]] ↔ [[extreme-weather-forecasting]]

## 2026-07-21 Lint 修复

### 幻觉修正
- source-rivermamba.md: Zigzag 曲线虽被研究但未用于最终模型 → 改为 Sweep 和 Gilbert；无法验证的 GloFAS F1 对比数字（0.4248 vs 0.1318）→ 替换为 Table 1 整体平均值（0.4589 vs 0.3582）
- rivermamba.md: GloFAS 性能数字（R² 0.8287/KGE 0.8837/F1 0.4248）无法在主论文 Table 1 验证 → 替换为 Table 1 可验证的整体平均值（R² 0.8728/KGE 0.9125/F1 0.4589）
- log.md: 同上 Zigzag + 数字修正

### 置信度修正
- source-rivermamba.md: confidence:high + source_count:1 违规 → source_count: 0, confidence: low
