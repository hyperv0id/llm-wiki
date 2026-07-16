# Ingest 报告：UrbanDiT — 正式 PDF 入库（2026-07-16）

基于完整论文 PDF（NeurIPS 2025 camera-ready）完成正式入库。原始 ingest 已于 2026-05-28 完成（source-summary + entity + unified-prompt-learning），本次为 PDF 物理文件归档及交叉引用完整性验证。

## 完成的操作
- **raw/ 入库**：拷贝 PDF → `raw/urbandit-diffusion-transformers-open-world-spatiotemporal-foundation-models.pdf`
- **交叉引用验证**：确认 `urbandit` ↔ `unified-prompt-learning` ↔ `rectified-flow` ↔ `spatio-temporal-foundation-model` 双向链接完整
- **index.md 一致性**：source-summary 和 entity 两条目均存在
- **frontmatter 更新**：source-urbandit.md、urbandit.md 的 `last_updated` → 2026-07-16

## 修改
- wiki/source-urbandit.md — last_updated: 2026-06-01 → 2026-07-16
- wiki/urbandit.md — last_updated: 2026-06-01 → 2026-07-16

## 未修改
- wiki/unified-prompt-learning.md — 内容无变动，last_updated 保留 2026-05-28
- wiki/spatio-temporal-foundation-model.md — 已有完整 UrbanDiT 条目及引用
- wiki/rectified-flow.md — 已有 UrbanDiT/InstaFlow 小节

## 新建交叉链接
无 — 已有交叉链接均完整。

## 时间线
1. 2026-05-12：基于摘要 + GitHub README 的初步 ingest
2. 2026-05-28：收到完整论文 PDF，全面重写 source-summary 和 entity 页面，创建 unified-prompt-learning 技术页
3. 2026-07-16：PDF 正式入库 raw/，交叉引用完整性验证，frontmatter 日期更新
