# Ingest 报告：PatchTST (Nie et al., ICLR 2023)

## 创建
- wiki/source-patchtst.md — WHY：ICLR 2023 里程碑论文，首次系统引入 patch tokenization + channel independence，回击"Transformer 无用论"，为后续模型提供核心设计范式
- wiki/patchtst.md — WHY：PatchTST 是 LSTF 领域的关键实体，其 patch+CI 设计被 SimDiff/CVPE/SparseTSF/CycleNet 等继承，需独立实体页面记录架构、性能、历史地位

## 修改
- wiki/patch-based-tokenization.md — WHY：PatchTST 是 patch tokenization 在时序 Transformer 中的首次系统化应用，需补充其具体参数设计（P=16, S=8）和三重收益
- wiki/channel-independence.md — WHY：PatchTST 是首个将 CI 引入 Transformer 的模型，需记录其在 CI 中的开创地位和消融证据
- wiki/instance-normalization.md — WHY：PatchTST 使用 RevIN 与 CI 配合，需补充具体使用方式
- wiki/informer.md — WHY：PatchTST 是 Informer 之后的重要继承模型，21% MSE 降幅需记录在历史意义中
- wiki/lstf.md — WHY：PatchTST 是 LSTF 进化链中的关键环节（tokenization rethink），且是对 Linear Model Challenge 的直接回应
- wiki/tslib.md — WHY：TSLib 将 PatchTST 作为 patch-wise Transformer 代表进行 benchmark
- wiki/simdiff.md — WHY：SimDiff 继承 PatchTST 的 patch+CI 设计扩展至扩散框架

## 新建交叉链接
- [[patchtst]] ↔ [[patch-based-tokenization]]
- [[patchtst]] ↔ [[channel-independence]]
- [[patchtst]] ↔ [[instance-normalization]]
- [[patchtst]] ↔ [[informer]]
- [[patchtst]] ↔ [[lstf]]
- [[patchtst]] ↔ [[simdiff]]
- [[patchtst]] ↔ [[cvpe]]
- [[patchtst]] ↔ [[crossformer]]
- [[patchtst]] ↔ [[sparsetsf]]
- [[patchtst]] ↔ [[tslib]]
