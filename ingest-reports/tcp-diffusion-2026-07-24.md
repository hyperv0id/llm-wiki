# Ingest 报告：TCP-Diffusion (2026-07-24)

## 创建
- wiki/source-tcp.md — WHY：论文 source-summary 页，300-500 字覆盖核心贡献（ARP、多模态编码器、NWP 集成）和关键结果
- wiki/tcp-diffusion.md — WHY：首个 DL 全球 TC 降水预测模型，entity 类型记录模型架构、性能与局限
- wiki/tropical-cyclone-precipitation-forecasting.md — WHY：TC 降水预测作为独立概念，与常规降水预测的差异和方法论
- wiki/adjacent-residual-prediction.md — WHY：ARP 是核心技术创新，从 NWP 残差预测理念移植到扩散模型框架

## 修改
- wiki/precipitation-nowcasting.md — WHY：新增 TC 降水预测章节和 TCP-Diffusion 引用，source_count 3→4
- wiki/diffusion-models.md — WHY：在应用领域添加 TC 降水预测案例，source_count 11→12
- wiki/extreme-weather-forecasting.md — WHY：在单一事件类型方法中加入 TC 降水预测，source_count 5→6
- wiki/index.md — WHY：添加 4 个新页面条目到 Sources/Entities/Concepts/Techniques 分类
- wiki/log.md — WHY：记录本次 ingest 操作

## 新建交叉链接
- [[tcp-diffusion]] ↔ [[source-tcp]]
- [[adjacent-residual-prediction]] ↔ [[tcp-diffusion]]
- [[tropical-cyclone-precipitation-forecasting]] ↔ [[tcp-diffusion]]
- [[tcp-diffusion]] ↔ [[precipitation-nowcasting]]
- [[tcp-diffusion]] ↔ [[diffusion-models]]
- [[tcp-diffusion]] ↔ [[extreme-weather-forecasting]]
- [[adjacent-residual-prediction]] ↔ [[precipitation-nowcasting]]

## Lint (2026-07-25)

### 严重（已修复）
- [x] tcp-diffusion.md / precipitation-nowcasting.md — ETS-24 "PreDiff 0.106" 幻觉：PDF Table 1 中 0.10587 是 U-Net，PreDiff 实际为 0.11931。修正为 0.119，提升率 ~39%→~23%

### 警告（已修复）
- [x] adjacent-residual-prediction.md — TCP-Diffusion 提及缺 wikilink → [[tcp-diffusion]]
- [x] tropical-cyclone-precipitation-forecasting.md — TCP-Diffusion ×2 + precipitation nowcasting 缺 wikilink
- [x] tcp-diffusion.md — ARP 提及缺 wikilink → [[adjacent-residual-prediction]]

### 幻觉交叉验证
其余全部 claim（作者/会议/方法/数据集/指标/消融/局限）与 pdftotext 提取的 PDF 原文一致。

### 仍存风险
- source-tcp.md: source_count:0 + confidence:low，待被其他源引用后升级
- adjacent-residual-prediction.md 中 GenCast/GraphCast 命名来自对 PDF 作者-年份引用的推断
