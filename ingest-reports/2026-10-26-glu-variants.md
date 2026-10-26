# Ingest 报告：GLU Variants Improve Transformer（Shazeer, arXiv:2002.05202v1）

通读范围：§1–§4、式 1–6、表 1–4 连同各表列头与标准差行、脚注 1–3；PDF 共 5 页，正文到表 4 结束，其后仅参考文献，无附录。源档 `raw/glu-variants-improve-transformer-2002-05202.pdf`。

## 创建
- wiki/swiglu.md — WHY：仓库原有 [[glu-gated-linear-unit]] 只有含偏置的 GLU 定义与 Mamba 输入门对照，[[gated-linear-units]] 只有 FaST 的 expert 复用写法，没有任何页承载 SwiGLU/ReGLU/GEGLU/Bilinear 的变体谱系、逐阶段前向数据流与三矩阵宽度折算；新建技术页补这一空位。
- wiki/source-shazeer-2020-glu-variants.md — WHY：单源摘要页，覆盖贡献（变体族 + 等参数对照）、T5-base 实验数字与论文自述边界。

## 修改
- wiki/glu-gated-linear-unit.md — WHY：补「Transformer FFN 中的无偏置形式与变体」一节（式 6 的三矩阵映射、$d_{ff}$ 折算、SQuAD 反例指向），并加 [[swiglu]] 链接；source_count 1→2。
- wiki/gated-linear-units.md — WHY：`与其他门控机制的对比` 表中的 SwiGLU 形式把门控激活记成 σ、三矩阵排成连乘，按原始定义更正并留「历史论断」；顺带加 [[swiglu]] 链接；source_count 1→2。
- wiki/transformer-architecture.md — WHY：Stage 4 的 ReLU 前馈正是该变体的替换位点，补一句折算与结论指向，使该页 FFN 小节可外链到变体页；source_count 1→2。
- wiki/index.md — WHY：Sources 段按字母序插入新 source 页（shaw → shazeer → shortcut），Techniques 段插入 swiglu（structural-token-sorting → swiglu → task-adaptive-dynamic-prompting）。

## 新建交叉链接
- [[swiglu]] ↔ [[glu-gated-linear-unit]]、[[gated-linear-units]]、[[transformer-architecture]]
- [[source-shazeer-2020-glu-variants]] ↔ [[swiglu]]

## 数值备注
- pdftotext 把式 6 下方「reduce … by a factor of 3/2」抽成「a factor of 32」；由 §3.1 的 3072→2048 判定原文为 $3/2$，页面按 $d_{ff}\to\tfrac{2}{3}d_{ff}$ 书写。
- §3.1 原文「which have thee weight matrices」是 three 的拼写错误，按三矩阵理解。
- 表 1 的 65,536 步列带 4 次运行的标准差，524,288 步列为单次运行；表 2–4 的标准差行取自 Raffel et al. 的多次运行，不覆盖本节各变体，页面已写明。
- 表 4 中 GLU 的 EM/F1（82.88/90.69）低于 ReLU 基线（83.18/90.87），与 §3.3「新变体在多数任务上最好」的概括并存；两处并列表述，不替论文取舍。
- 表 1 的 65,536 步档排序与 524,288 步档一致，但 SwiGLU 的标准差（0.010）为全表最大，页面按原表数值记录、未做合并。
