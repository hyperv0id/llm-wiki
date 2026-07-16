## 2026-07-24 补全
- **raw/ 新增 PDF**：`raw/tft-temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting.pdf`（与已有 `raw/1912.09363.pdf` 并存）
- **Lint 修复**：source-tft、gated-residual-network、variable-selection-network、interpretable-multi-head-attention 的 confidence high→medium（source_count:1 不合规）

# Ingest 报告：TFT / arXiv:1912.09363

## 创建
- wiki/source-tft.md — WHY：source-summary，slug `src-tft`；记录多 horizon 输入分类、GRN/变量选择/可解释注意力、实验与局限
- wiki/tft.md — WHY：实体页面，架构、分位数输出、可解释用例与历史定位
- wiki/gated-residual-network.md — WHY：TFT 核心 GRN+GLU 构建块技术页
- wiki/variable-selection-network.md — WHY：实例级变量选择与全局重要性分析技术页
- wiki/interpretable-multi-head-attention.md — WHY：共享 value 的可解释多头注意力技术页

## 修改
- wiki/index.md — WHY：Sources / Entities / Techniques 登记 TFT 相关页面
- wiki/log.md — WHY：追加 ingest 日志
- wiki/direct-forecast.md — WHY：将 TFT 记为早期 direct multi-horizon + quantile 代表
- wiki/heterogeneous-covariates.md — WHY：表格中的 TFT 改为 wikilink，相关概念补链
- wiki/source-nbeatsx.md / wiki/nbeatsx.md — WHY：外生谱系上标注 TFT 前驱
- wiki/source-tide.md / wiki/tide.md — WHY：协变量感知前驱链接
- wiki/source-timexer.md — WHY：Transformer 外生谱系补 TFT
- wiki/source-exost.md — WHY：外生相关工作交叉链接 TFT
- wiki/glu-gated-linear-unit.md — WHY：时序侧 GRN 用法交叉链接

## 新建交叉链接
- [[source-tft]] ↔ [[tft]]
- [[tft]] ↔ [[gated-residual-network]] / [[variable-selection-network]] / [[interpretable-multi-head-attention]]
- [[tft]] ↔ [[direct-forecast]] / [[heterogeneous-covariates]] / [[nbeatsx]] / [[tide]] / [[source-timexer]]
- [[gated-residual-network]] ↔ [[glu-gated-linear-unit]]

## 源文件
- 仓库内：`raw/1912.09363.pdf`（不可变，已存在）
- 外部任务路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/1912.09363.pdf`

## 2026-07-24 Lint 复查

对照 PDF (pdftotext) 完整执行 CLAUDE.md Lint 检查清单。

### 修复
- wiki/source-tft.md：补充缺失的 `created: 2026-07-13` 字段
- wiki/tft.md：Historical Position 段落添加编辑性定位 callout（跨模型比较基于 wiki 知识，非全部来自 TFT 论文）

### 验证通过
- 幻觉检查：作者、年份、方法名、数据集配置、性能/消融数字、可解释用例全部与 PDF 原文一致
- 结构：所有页面 frontmatter 齐全、type 合法
- 引用：所有 [^src-tft] 脚注定义存在，source_count 一致
- 链接：全部 wikilink 存在，交叉引用完整
- 时效/置信：source_count:1 + confidence:medium 合规
