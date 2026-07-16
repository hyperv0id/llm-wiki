# Ingest 报告：TimeGrad gap-filling pass (2026-07-25)

## 背景

TimeGrad (Rasul et al., ICML 2021) 已于 2026-05-31 完成初始 ingest（source-summary + entity）。本次对照用户提供的 PDF（md5=b093d9fe，与 raw/2101.12072.pdf 一致）完整验证内容准确性，并补全交叉引用缺口。

## 创建
- [[crps]] — WHY：CRPS 是 TimeGrad Section 4.1 详细定义的核心评估指标，wiki 此前缺失该概念页。TimeGrad 及后续大量扩散/概率时序工作（DiffSTG、CSDI 等）均使用 CRPS_sum 评估。

## 修改
- [[source-timegrad]] — WHY：刷新 last_updated 标记本轮验证通过；Lint 后发现自引用循环（17+ 处 [^src-timegrad]），移除全部自引用，source_count: 1→0, confidence: medium→low（仓库惯例：source-summary 不自引）
- [[timegrad]] — WHY：刷新 last_updated；Lint 中修复 §4 局限性引用范围（CSDI 可学习调度非 TimeGrad 内容）和关联页面格式
- [[langevin-dynamics]] — WHY：TimeGrad Section 2 明确描述推理过程为退火 Langevin 动力学，添加"时间序列扩散模型中的应用"小节，source_count 1→2
- [[energy-based-model]] — WHY：TimeGrad Introduction 自称"autoregressive EBMs"，添加交叉引用，source_count 1→2
- [[index]] — WHY：新增 crps 条目；所有已修改页面的交叉链接均完整

## 新建交叉链接
- [[langevin-dynamics]] ↔ [[timegrad]] — 退火 Langevin 采样在时序扩散条件生成中的首次应用
- [[energy-based-model]] ↔ [[timegrad]] — TimeGrad 作为首个自回归 EBM 时序预测方法
- [[crps]] ↔ [[timegrad]] — CRPS 的定义和 CRPS_sum 变体均引用 TimeGrad
- [[crps]] ↔ [[csdi]] — 间接关联（CSDI 使用 CRPS 评估插补质量）

## 已验证
- pdftotext 全文与前次 ingest 的 source-summary 和 entity 内容一致，无幻觉、无错引
- 方法描述（2层 LSTM h=40, 8残差块 GAU, N=100, β∈[1e-4,0.1], S=100 采样, CRPS_sum 公式）与原文完全匹配
- 实验结果（Table 2 中 5/6 数据集第一）与原文一致
- 消融结论（N≈10 即接近最优）与 Figure 3 一致

## 未创建
- mean-scaling.md — 均值缩放的核心概念源于 DeepAR，TimeGrad 仅继承使用。待 DeepAR 专项 ingest 时创建。
- ebm-time-series.md — 当前 wiki 中 [[energy-based-model]] 和 [[timegrad]] 已充分覆盖该交叉领域。

## Lint 后修复 (2026-07-25)

对照 CLAUDE.md 检查清单全项 + pdftotext 幻觉检查。

### 严重（已修复）
- [x] source-timegrad.md — **自引用循环**：17+ 处 [^src-timegrad] 内联引用 + 脚注定义形成自引用回路。移除全部自引用，source_count: 1→0，confidence: medium→low（仓库惯例：source-summary 不自引，参见 ProbTS/STPDE/QDF/StormInsight 前例）

### 警告（已修复）
- [x] timegrad.md — **局限性 §4 引用不精确**："CSDI 后来将其改进为可学习调度" 带 [^src-timegrad]，但 TimeGrad 不可能讨论 CSDI 后继改进。已将 [^src-timegrad] 缩限至"所有数据集使用相同 β 调度"前
- [x] timegrad.md — **关联页面格式**：crps-autoregressive-finetuning 条目缺少列表符号，已补

### 幻觉交叉验证通过
全部方法描述（2层 LSTM h=40, 8块 GAU, N=100, β∈[1e-4,0.1], S=100, CRPS_sum）、实验结果（5/6 第一, Wikipedia 0.0485 vs MAF 0.063）、消融结论（N≈10）与 PDF 原文逐条验证一致，无捏造或错引。

### 已验证
- crps.md: source_count:1, confidence:low（单源概念页）——合规
- timegrad.md: source_count:9, confidence:medium（主论断单源 + 8 关联源）——合规
- langevin-dynamics.md: source_count:2, confidence:medium——合规
- energy-based-model.md: source_count:2, confidence:low——合规
- 全部 wikilink 存在，无断链，交叉引用双向
- 全部新页面/修改页面在 index.md 对应类别登记

### 仍存风险
- source-timegrad.md: source_count:0, confidence:low，待被其他源引用后升级
- crps.md: source_count:1, confidence:low，同为单源概念页
