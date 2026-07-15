# Ingest 报告：OmniField — Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning

## 创建
- wiki/source-omnifield.md — WHY：ICLR 2026 论文，连续性感知多模态条件化神经场框架，是 CNF 在科学多模态时空数据上的重要里程碑
- wiki/omnifield.md — WHY：模型实体页，编码器-处理器-解码器架构统一四类任务（重建/插值/预测/跨模态预测）
- wiki/multimodal-crosstalk.md — WHY：MCT 块是跨模态信息交换的核心新机制，通过全局特征 z 实现轻量级跨模态条件化
- wiki/iterative-cross-modal-refinement.md — WHY：ICMR 是噪声鲁棒性的关键创新，迭代 MCT + 池化桥接逐步对齐异构模态信号
- wiki/fleximodal-fusion.md — WHY：presence-mask 门控机制让单模型适应任意输入子集，优于传统的 ModDrop 训练增强

## 修改
- wiki/index.md — WHY：新增 source/entity/technique 条目
- wiki/log.md — WHY：记录 ingest 操作

## 新建交叉链接
- [[omnifield]] ↔ [[multimodal-crosstalk]] ↔ [[iterative-cross-modal-refinement]] ↔ [[fleximodal-fusion]]
- [[source-omnifield]] → [[omnifield]]
- 这些页面的反向链接均在 source-summary 和各 technique 页面的"相关"section 中建立

## 备注
- 论文主体（方法/实验/结论）通过 arXiv HTML 版 (2511.02205) 获取完整内容，因 PDF 使用复杂嵌入式字体导致 pdftotext 仅提取了约 20% 内容
- SCENT、MIA、PROSE-FD、CORAL 等基线模型尚未在 wiki 中有独立页面（留待后续 ingest）
- 论文贡献了 ClimSim-LHW 和 ML-ready EPA-AQS 数据集，可在后续创建数据集相关页面时引用
