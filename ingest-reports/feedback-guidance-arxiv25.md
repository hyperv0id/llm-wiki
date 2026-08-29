# Ingest 报告：Feedback Guidance of Diffusion Models (FBG)

**源文件**: `raw/feedback-guidance-diffusion-models-arxiv25.pdf`（arXiv:2506.06085 v2，NeurIPS 2025 poster，Koulischer et al.，Ghent University - imec / Radboud Donders Institute）
**日期**: 2026-08-29

## 背景

上轮查询（2026-08-29）盘点 FENCE（AAAI 2026）的 31 条参考文献时，确认 FBG 是其中关联性最强的未收录引用：FENCE 的动态引导尺度公式与后验追踪机制均标注采用自该工作，但 wiki 的 [[feedback-diffusion-guidance]] 技术页此前仅基于 FENCE 单源转述（source_count: 1）。本次 ingest 补齐原始出处。

## 创建

- `wiki/source-feedback-guidance-diffusion-models-arxiv25.md` — WHY：新源文件的 source-summary，记录核心论点（加性误差假设导出动态引导尺度）、作者报告的实验结果（ImageNet512 EDM2-XS / SDv2 T2I）与论文自述局限，并单列与 FENCE 的公式/超参差异。

## 修改

- `wiki/feedback-diffusion-guidance.md` — WHY：原页面完全建立在 FENCE 转述之上；以原文为主来源重写：新增误差模型视角（加性 vs 乘性）、原文公式（含 p_θ(c) 归一化，并加 callout 标注与 FENCE 简写形式的差异）、后验追踪细节（自我参照偏差、t0/t1 重参数化）、开环/闭环定位、完整实验证据与论文自述局限；FENCE 相关内容收敛到"与 FENCE 的关系"一节。source_count 1→2。
- `wiki/fence.md` — WHY：正文"理论公式采用自 Koulischer et al. 2025"处补内联引用与 wikilink；对比表增加 FBG 行并附说明；另见节链接 FBG 技术页与 source 页。source_count 2→3。
- `wiki/classifier-free-guidance.md` — WHY："动态 CFG（反馈引导）"节原先仅引 FENCE，补 FBG 原始出处引用与原文公式形式说明。source_count 7→8。
- `wiki/index.md` — WHY：Sources 节登记新 source 页（置于 source-fence 旁），Techniques 节更新 feedback-diffusion-guidance 条目描述（AAAI 2026 → NeurIPS 2025 / AAAI 2026 双归属）。

## 新建交叉链接

- [[source-feedback-guidance-diffusion-models-arxiv25]] ↔ [[feedback-diffusion-guidance]]
- [[source-feedback-guidance-diffusion-models-arxiv25]] ↔ [[fence]]
- [[source-feedback-guidance-diffusion-models-arxiv25]] ↔ [[classifier-free-guidance]]

## 口径说明

- FBG 的 venue（NeurIPS 2025 poster）经 arXiv abs 页 comments 字段核实（上轮查询时未核实，本次已补）。
- 上轮查询中提到的"Feedback Guidance 后续发表状态未独立核实"现已解决。
- 表 1 数值从 PDF 提取时列序有扰动，wiki 中仅引用可交叉验证的数字（FID 3.76 vs 5.00 等与图 3/图 8 标注一致者）。
- LIG、CFG++、autoguidance、Dynamic Negative Guidance 等被引工作未建页（避免页面扩散）；如后续 ingest LIG（Kynkääniemi et al., NeurIPS 2024）可再补。
