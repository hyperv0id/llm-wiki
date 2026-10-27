# Ingest 报告：shazeer-2017-sparse-moe-1701-06538.pdf

## 创建
- wiki/sparse-gated-moe-layer.md — WHY：论文的 technique 本体（noisy top-k 门控、importance/load 双均衡损失、混合并行训练、两级 hierarchical MoE）。库内 `mixture-of-experts.md` 以现代 MoE 为主线且引用混杂，不宜承载 2017 原始口径，按任务定位新开「源头」页
- wiki/source-shazeer-2017-sparse-moe.md — WHY：一份源一个 source-summary，schema 要求

## 修改
- wiki/mixture-of-experts.md — WHY：仅末尾加一行谱系句指向新页并补脚注定义，source_count 16→17；按任务要求不大改
- wiki/index.md — WHY：Sources 按字母序插入（shazeer-2017 先于 shazeer-2019/2020），Techniques 新条目（sparse-balanced 之后、sparse-teacher 之前）
- wiki/log.md — WHY：追加一条 ingest 记录

## 新建交叉链接
- [[sparse-gated-moe-layer]] ↔ [[mixture-of-experts]]（新页谱系段 + 老页「谱系源头」段互指）
- [[source-shazeer-2017-sparse-moe]] ↔ [[sparse-gated-moe-layer]]

## 数值备注
- 任务提示所称「消融（k=1/2/4、有无噪声）」与「train speedup」列在论文中不存在：唯一消融是 Table 6（$w_{\text{importance}}$/$w_{\text{load}}$ 六种组合，含双 0 对照）；噪声的作用经由 load 损失（式 9 依赖噪声分布）间接检验。训练开销以 Table 1–5 的 training time（天/GPU 数）与 TFLOPS/GPU 列呈现，无 speedup 比值列。
- 「137B/512 expert」对应关系：137B 是 Table 8 的 MoE-131072-h（131072 expert、137577.6M 参数）；512 expert 是 Table 5 多语模型（非分级 $n=512$、$k=2$、102M ops）。两表均无 BLEU 之外的 speedup 列，BLEU 在 Table 2–4（2048 expert 模型）。
- Table 5 GNMT-Multi 硬件栏 PDF 原文即「21 days, 96 k20s」，-layout 与 raw 两种提取一致；「k20s」疑为「k80」排印误，页面只写「21 天/96 GPU」不写型号。
- 附录 C.2 正文称 MoE-143M 比已发表最优低 18%；按 Table 1 的 28.0/34.7 实算为 19.3%。页面照录论文口径并并列两个原始数。
- §3.1 单设备 expert batch 公式 pdftotext 读作「approximately kb/n b examples」（含一多余 b）；多设备处为 $kbd/n$ 且正文自洽于 "factor of d improvement"。页面取 $kb/n$（单设备）与 $kbP/n$（合并后）。
- Figure 2/3/4 为曲线图，图内逐点数值不进文本层；正文汇总数（24%、39%、−19%）按论文正文口径引用，并回打了对应 Table 7/8/5 的端点值。
- 论文版本：arXiv:1701.06538v1（2017-01-23），ICLR 2017「Under review as a conference paper」版式，仅此一版。
