# Ingest 报告：ProbTS（补充 ingest）

**原始 ingest**: 2026-07-13  
**本次操作**: 2026-07-16 — 补充 raw PDF 副本与交叉验证

**Source PDF (immutable)**:
- `raw/probts-benchmarking-point-and-distributional-forecasting.pdf`（本次新增，kebab-case 命名）
- `raw/2310.07446.pdf`（初始 ingest，arXiv ID 命名，保留不动）

**Slug**: `src-probts`  
**Venue**: NeurIPS 2024 Datasets and Benchmarks | arXiv:2310.07446v5

---

## 已有页面（初始 ingest 创建，本次仅验证）

- `wiki/source-probts.md` — source-summary，核心论点/发现/贡献/局限均已覆盖。
- `wiki/probts.md` — 基准工具实体，设计目标/模块/覆盖模型/结论均有 [^src-probts] 引用。
- `wiki/ar-vs-nar-decoding.md` — 论文核心方法轴概念页，3 源引用。
- `wiki/non-gaussianity.md` — 窗口分布复杂度量化指标概念页。

## 已有修改（初始 ingest 完成，本次仅验证无需再改）

- `wiki/index.md` — 已登记。
- `wiki/instance-normalization.md` — RevIN vs mean scaling 跨场景结论已补。
- `wiki/generative-time-series-forecasting.md` — 长程概率预测开放问题已补。
- `wiki/timegrad.md` / `wiki/csdi.md` / `wiki/patchtst.md` / `wiki/timesfm.md` / `wiki/chronos.md` — 反向链接已加。

## 本次修改

- `wiki/source-probts.md` — 添加新 raw PDF 路径 `raw/probts-benchmarking-point-and-distributional-forecasting.pdf`；更新 `last_updated: 2026-07-16`。
- `wiki/probts.md` — 更新 `last_updated: 2026-07-16`。

## 交叉验证结论

对 source-probts.md 的五项主要发现与论文 §4.1–§4.2 逐条对照验证，全部准确：

| 发现 | 论文出处 | 状态 |
|------|----------|------|
| 定制长程架构在短程失效 | §4.1 Fig 2c–2d (Solar-S) | ✓ |
| 既有概率模型长程分布预测崩溃 | §4.1 (TimeGrad on ETTm1-L, CSDI 显存/效率) | ✓ |
| AR 在强季节性上可反超 | §4.1 Fig 3c–3d (Traffic) | ✓ |
| RevIN 主要利好长程 AR | §4.1 Fig 3b (ETTh1 GRU-NVP+RevIN) | ✓ |
| TSFM 复现 AR 长程劣势与复杂分布短板 | §4.2 Fig 4a–4b | ✓ |

## 新建交叉链接

无新增。初始交叉链接已完整覆盖。

## 未创建（有意收窄）

无新增页面。所有页面的 [^src-probts] 引用与 `source_count` 均与实际一致。

---

## 2026-07-16 Lint 修复

### 严重（已修复）
- [x] `wiki/source-probts.md` — 移除 15 处 `[^src-probts]` 自引用 + 脚注定义（source-summary 不应自引用），`source_count: 1→0`，`confidence: medium→low`
- [x] `wiki/source-probts.md` — `[[generative-style-decoder|NAR]]` → `[[ar-vs-nar-decoding|NAR]]`（ProbTS 讨论通用 AR/NAR 解码，非 Informer 专用 Generative Style Decoder）

### 幻觉检查
对照 PDF (pdftotext) 逐条验证：作者/机构/会议/arXiv、AR/NAR 公式、分布头三类、FT/FS/非高斯性 JS 公式、STL 分解、窗口长度（短 30/长 336）、100 采样、五项主要发现与 §4.1–§4.2 对照——全部与 PDF 原文一致。

### 仍存风险
- source-probts.md source_count:0 + confidence:low：source-summary 无交叉来源验证，待该论文被其他源引用后升级 confidence
