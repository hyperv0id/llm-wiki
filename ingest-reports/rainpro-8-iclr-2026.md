# Ingest 报告：rainpro-8-rainfall-probability-iclr-2026

## 创建
- wiki/source-rainpro.md — WHY：论文 source-summary，覆盖 RainPro-8 核心贡献、结果和局限性
- wiki/rainpro.md — WHY：RainPro 模型系列实体（RainPro-8/8R/2R），多源融合概率降水预报
- wiki/metnet.md — WHY：MetNet 系列实体，RainPro 的主要参照和对照基线，论文中大量讨论
- wiki/ordinal-consistent-loss.md — WHY：有序一致性损失技术，条件概率链式法则保证降水强度单调性，首次应用于降水预报

## 修改
- wiki/precipitation-nowcasting.md — WHY：从单源（QCGS）扩展到双源，增加多源数据融合路线（8h 级别）、MetNet 系列、RainPro 创新点
- wiki/index.md — WHY：新增 source/entity/technique 条目，更新 precipitation-nowcasting 描述

## 新建交叉链接
- [[precipitation-nowcasting]] ↔ [[rainpro]]
- [[precipitation-nowcasting]] ↔ [[metnet]]
- [[precipitation-nowcasting]] ↔ [[ordinal-consistent-loss]]
- [[rainpro]] ↔ [[metnet]]
- [[rainpro]] ↔ [[ordinal-consistent-loss]]
- [[rainpro]] ↔ [[source-rainpro]]
- [[metnet]] ↔ [[source-rainpro]]
- [[ordinal-consistent-loss]] ↔ [[source-rainpro]]
- [[precipitation-nowcasting]] → [[extreme-weather-forecasting]]（已有，补充 RainPro 概率输出辅助极端降水预警）

## Lint — 2026-07-21

### 严重（已修复）
- [x] wiki/rainpro.md — 训练时间幻觉："约 13.6 小时" 原文为 "approximately 13 hours"，改为 "约 13 小时"

### 警告（已修复）
- [x] wiki/ordinal-consistent-loss.md — 缺失引用：Fernandes & Cardoso (2018) 句末缺 [^src-rainpro]

### 幻觉交叉验证通过
对照 PDF (pdftotext) 逐条验证：作者（6 人）/机构（AU & Cordulus）/ICLR 2026/CRPS 0.06096/CSI 0.279/FSS 0.537/36.7M 参数/227M MetNet-3/EUMETSAT 11 通道（VIS0.6~IR13.4）/GFS 122 变量/Copernicus DEM/H100 ~13h/α=10/Ordinal Consistent Loss 公式/Fernandes & Cardoso (2018)/SEVIR + DiffCast 13×/Integrated Gradients/MetNet-3* 忠实现复/3 种子鲁棒性/65% 超越 NWP——除训练时间 13h→13.6h 笔误外，全部与 PDF 原文一致。

### 仍存风险
- 无。本轮 ingest 仅 1 个源文件，所有新建页面 source_count≤1，符合单源置信度策略。
