# Ingest 报告：STGformer (arXiv:2410.00385v2)

源文件：raw/stgformer-nie-2024.pdf（+ 同名 .txt）。日期：2026-08-30。

## 创建
- wiki/source-stgformer.md — WHY：每份 raw 源文件一个 source-summary 页，归档论文核心主张、论文报告的数字与口径注记（CA 全图仅效率测量、跨年非跨网、无 venue）。
- wiki/stgformer.md — WHY：模型实体页，作为「Scaling 时空预测模型设计」计划中效率锚点（100×/99.8%@8600 节点）与同图跨年泛化证据的挂靠点。
- wiki/stg-attention.md — WHY：论文核心机制（图传播多阶 × 统一时空共享 QKV × 分解内积线性化 × 递归门控交互）值得独立技术页，供线性注意力谱系与大规模建模页引用。

## 修改
- wiki/staeformer.md — WHY：STGformer 沿用其数据嵌入（含 Xste）并以之为效率/精度对照，补后续工作关系与论文报告数字；source_count 3→4。
- wiki/large-scale-spatial-temporal-graph.md — WHY：「无结构方法」类目补 STGformer 效率证据（8600 节点 100× 加速 / 99.8% 显存降 / FLOPs 比值 0.00131）；source_count 6→7。
- wiki/spatio-temporal-ood-learning.md — WHY：补同图跨年（T-OOD）模型侧证据，并与 ST-OOD 基准的 OUT 结论按各自协议分立记录；source_count 4→5。
- wiki/traffic-forecasting.md — WHY：「Large-Scale Linear-Complexity Modeling」节补 STGformer 条目；source_count 52→53。
- wiki/linear-attention-unified-framework.md — WHY：补交通域实例（分解内积线性注意力 × 图传播，O(N+T)）；source_count 1→2。
- wiki/index.md — WHY：收录 3 个新页面（Sources/Entities/Techniques 末尾追加）。
- wiki/log.md — WHY：按仅追加原则记录本次 ingest。

## 新建交叉链接
- [[stgformer]] ↔ [[staeformer]]
- [[stgformer]] ↔ [[large-scale-spatial-temporal-graph]]
- [[stgformer]] ↔ [[spatio-temporal-ood-learning]]
- [[stgformer]] ↔ [[traffic-forecasting]]
- [[stg-attention]] ↔ [[linear-attention-unified-framework]]
- [[stg-attention]] ↔ [[adaptive-graph-agent-attention]]、[[query-aggregate-attention]]（同问题域对照，本 wiki 归类）

## 备注
- raw 文件名后缀 "nie" 为下载时误标（实际一作 Hongjun Wang）；遵守 raw/ 不可变策略未重命名，已在 log.md 注记，著录以 PDF 水印与作者栏为准。
- STGformer 与 Tong Nie 等的同名工作（另一篇 ST 论文）易混淆；本页仅覆盖 arXiv:2410.00385。

## 审查补记（2026-08-30，提交前自查）

- 严重 1：「Table I 全面优于全部对比基线」与「跨年 Table II 三子集均最优」超出论文范围。逐格复核后修正为：Table I 平均 MAE 三子集均为对比模型最低 + 全部指标一致优于 STAEformer（论文自述），SD 平均 RMSE 被 D2STGNN 微弱超过（29.51 vs 29.52）、BA 平均 MAPE 被 D2STGNN 超过（15.04% vs 15.22%）；Table II 平均 MAE 三子集均最低，平均 RMSE/MAPE 在 SD（STID MAPE）/BA（STID）/LA（GWNET）被超过，页面列明例外格。PEMS Table III「4×3 平均格全部最低」逐格核实成立，保留。
- 警告 1：Xste「参数量随 N 变化」改为显式推断标注（STAEformer 原文未 ingest，STE 可学习性未在原文层核实）。
- 修复落点：source-stgformer.md、stgformer.md、spatio-temporal-ood-learning.md、traffic-forecasting.md（措辞同步）；staeformer.md 表述仅陈述事实未动。详见 log.md lint-fix 条目。

## 复核补记（2026-08-30，ingest 代理逐数字复核 PDF 后追加）

- 版本与著录核实：全题 STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting；水印 arXiv:2410.00385v2 [cs.LG] 15 Oct 2024；作者 Hongjun Wang、Jiyuan Chen、Tong Pan、Zheng Dong、Lingyu Zhang、Renhe Jiang、Xuan Song，通讯作者 Xuan Song，PDF 内无任何 "Nie" 署名；venue 不确定性维持（IEEE 模板占位页眉，无接收信息）。
- 警告 2（效率口径）：100× 加速与 99.8% GPU 内存降幅仅出现于摘要与引言贡献列表，正文实验章节（Sec V）无对应测量表/图与批量设置；效率的正文证据仅为 Sec IV-D 的 FLOPs 解析对比（0.00131/99.869%）。已在 stgformer.md「论文报告的数字」、stg-attention.md「论文报告的证据」、source-stgformer.md「论文报告的实验」注明出处限定。
- 修正 4（例外格补全）：审查补记的例外清单漏 BA 平均 RMSE（GWNET 33.41 vs STGformer 33.50），且未覆盖 per-horizon 例外（SD h3 MAE 14.97 vs D2STGNN 14.92、SD h3 RMSE 24.96 vs 24.95、SD h6 RMSE 29.26 vs 29.24；BA h3 MAPE 12.72% vs 12.12%、BA h6 MAPE 15.12% vs 14.89%，对手均为 D2STGNN）。已在 stgformer.md 与 source-stgformer.md 的例外注记中列全 Table I 全部 8 个例外格（LA 无例外格）。
- 修正 5（设置口径）：stgformer.md 范围注记中 12→12 步长明确归属 LargeST（论文原文），PEMS 的输入/输出步长论文未逐条写明。
- 备注更正：原备注「Tong Nie 等的同名工作（另一篇 ST 论文）」经 web 检索不能证实——未检索到 Tong Nie 的同名 ST 论文；同名工作检索到的是 3D 人体姿态 STGFormer（arXiv:2407.10099），Tong Nie 为同领域另一位研究者。文件名 "nie" 后缀的确切成因无法从 PDF 核实，按「疑为混淆误标」记录（source-stgformer.md 已按此口径注记）。
- 关联页处置：grep 确认 [[quest-attention]]、[[wire]] 均未提及 STGformer 且主题相关性弱（softmax 键归一化 / 谱旋转位置编码），未强行加链；staeformer、large-scale-spatial-temporal-graph、linear-attention-unified-framework、traffic-forecasting 四页的 STGformer 增补段逐句核对与 PDF 一致，未改动。
- 并行说明：本次复核期间另一审查进程对 stgformer.md、source-stgformer.md、spatio-temporal-ood-learning.md 做了同方向口径精修（含 Xste 推断标注），逐条对照 PDF 核实无误后保留。

