# Ingest 报告：Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation

源：arXiv:2108.12409v2（ICLR 2022 正式版，2022-04-22，25 页）。通读全文：摘要、§1–§2（含 §2.1 设置、§2.2 三种基线）、§3 方法、§4 结果（§4.1 WikiText-103/Books、§4.2 CC100+RoBERTa）、§5 相关工作、§6 结论、Appendix A（A.1 Table 1–4、A.2 Table 5/Fig 8、A.3 Books、A.4 Table 11/12）、Appendix B（滑窗与 early token curse，Table 13–15）。PDF 归档为 `raw/alibi-2108-12409.pdf`（`*.pdf` 被 `.gitignore` 排除，只保留本地副本；`pdftotext -layout` 文本 1492 行）。

## 创建
- `raw/alibi-2108-12409.pdf` — WHY：本轮之前 wiki 无原文副本，推导与表号无法回溯；按仓库惯例归档 PDF（不入 git）。

## 修改
- `wiki/alibi.md` — WHY：补模型类页必需的「数据流与维度变换」四阶段（投影 → 偏置矩阵 → logits+softmax → 加权求和，含 `\bcancel` 消维与 softmax 规约轴）；补斜率几何级的前提与距离尺度推导、外推机制与 Appendix B 边界、四方法对照表（Table 4）、Table 5 与 Table 1 数值；修正斜率序列写法与效率口径
- `wiki/source-alibi.md` — WHY：补官方 URL、版本（v2/ICLR 2022）、代码与本地副本；摘要改为论点—方法—证据—局限四段，单源证据 confidence 设 medium，source_count 与正文实际引用数（1）一致
- `wiki/geometric-slope-schedule.md` — WHY：逐条补 §3/§4 出处；修正「$2^{-8/n},2^{-7/n},\dots,2^0$」为原文的起点=公比=$2^{-8/n}$、右端恒 $2^{-8}$；补序列两端固定与 1-nat 距离尺度推导，标明论文未给定理
- `wiki/position-extrapolation.md` — WHY：原「$2L$ 峰值→$>3L$ 下降」单曲线与原文两设置不符，改为 WikiText-103 与 CC100 分写并给 Table 2/4 与 Figure 6 数值；新增「收益来源的边界」；实际应用按摘要与 §4.2 改写为长输入推理、免长序列预训练、上下文示例与更长输出四条
- `wiki/context-window-extension.md` — WHY：ALiBi 行「~3L」改为两设置实测口径（WikiText-103 到 6× 仍改善、CC100 峰值约 $2L$），并补引用

## 新建交叉链接
- 无新建页面。[[alibi]] ↔ [[position-extrapolation]]、[[context-window-extension]]、[[geometric-slope-schedule]]、[[source-alibi]] 既有链接保持，未新增 dangling link；[[index]] 未改（两页标题与一行摘要均未变动）

## 数值备注
- `pdftotext` 丢失上下标负号：§3 的 8 头序列被提为 "211 , 212 , ..., 218"，16 头为 "20.5 , 21 , ..., 28"，公比说明同样丢负号。本页按 §3 的公式句（"starts at $2^{-8/n}$ and uses that same value as its ratio"）与 Figure 3 重构，未引入新数值
- 图 1、4、5、6、8、9、11 的坐标轴标题在提取文本中残缺（如 "Perplexi). ( ←)"、"Training W rds Per Sec nd"），图内数值未引用；所有数字改取 Table 1–15
- 摘要的「training 11% faster and using 11% less memory」对应 1.3B 模型与更长序列基线的同时长比较（Table 11：24.6 对 26.2 GB、5.5k 小时、9.30 对 9.24；§4.2 的 11% 由 5 个采样点估计）。Table 1 显示同 $L$ 下速度持平（$L=512$：28.3k 对 28.5k WPS）、显存相同、掩码多出 $n\times L\times L$（部分实验 ≤100 MB）。两处口径在页面与摘要里分别写明，不合并成单一数字
- Table 4 中 ALiBi 的最低点 17.15 出现在 $L_{valid}=13072$，16072 回升到 17.22；页面写区间 17.15–17.22，未写成单调下降
- Table 3 的 T5 bias 自 13024 起 OOM、Table 4 的自 14072 起 OOM，两表不同，页面只引用 Table 4 的 L=3072 组，未混用

## 矛盾处理
- 无跨源矛盾。`wiki/alibi.md` 与 `wiki/geometric-slope-schedule.md` 原先把 $n$ 头斜率写成 $2^{-8/n},2^{-7/n},\dots,2^0$，与原文「起点与公比同为 $2^{-8/n}$、右端恒 $2^{-8}$」不符；来源为同一篇论文（此前无原文副本），故按源更正、不加 `## 历史论断`，修正记录写入 `log.md` 的本条 ingest 与上面「修改」段。
- `alibi.md` 效率表的「+11%」与原文摘要一致，但只成立于 1.3B 同时长比较；同 $L$ 的 Table 1 为持平。按「两处口径分别归因」处理，未替论文合并数字。
