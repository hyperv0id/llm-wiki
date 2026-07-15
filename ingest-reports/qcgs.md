# Ingest 报告：QCGS (Station2Radar)

## 源文件
`raw/qcgs-query-conditioned-gaussian-splatting-precipitation-nowcasting.pdf`
arXiv:2603.00418, ICLR 2026, Kim et al. (KAIST)

## 创建
- `wiki/source-qcgs.md` — 核心论点、方法概要、关键结果、局限性（source-summary）
- `wiki/qcgs.md` — QCGS 完整技术页面：三阶段流水线（Radar Point Proposal + Rainfall-Aware Sampling + INR Gaussian Parameter Estimator）、数学公式、AWS 锚定策略
- `wiki/gaussian-splatting.md` — 3DGS→2DGS 技术演化、与传统高斯加权插值的等价性、与 INR 的关系
- `wiki/implicit-neural-representation.md` — INR 核心特性（分辨率无关/可微/密集查询瓶颈）、QCGS 中的应用
- `wiki/precipitation-nowcasting.md` — 降水临近预报概念页面：尺度不匹配挑战、方法演进（光流→ConvLSTM→扩散→无雷达）、QCGS 的定位

## 修改
- `wiki/extreme-weather-forecasting.md` — WHY：添加降水临近预报和 QCGS 的 cross-reference，更新 source_count 2→3、last_updated

## 新建交叉链接
- [[qcgs]] ↔ [[gaussian-splatting]] ↔ [[implicit-neural-representation]]
- [[qcgs]] ↔ [[precipitation-nowcasting]]
- [[precipitation-nowcasting]] → [[extreme-weather-forecasting]]
- [[qcgs]] → [[extreme-weather-forecasting]]

## 备注
- PDF 字体问题导致 pdftotext 仅提取出标题/摘要/引言开头，论文全文从 arXiv HTML 版 (2603.00418v1) 和 PaperNotes 分析中获取
- 论文未被引用的潜在 wiki 页面：ConvLSTM, PreDiff, Sat2Radar/NPM, IMERG, MSWEP, GSMaP, DGMR, CasCast, DiffCast, MetNet
