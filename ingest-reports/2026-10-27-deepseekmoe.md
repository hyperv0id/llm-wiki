# Ingest 报告：DeepSeekMoE (arXiv:2401.06066v1)

## 创建

- `wiki/deepseekmoe.md` — technique 页：两策略的公式、参数/激活记账式、分阶段数据流、双均衡损失、与 GShard/Shazeer 的机制对照表、2B/16B/145B 三档证据与边界。
- `wiki/source-dai-2024-deepseekmoe.md` — source-summary：问题（hybridity/redundancy）、两策略、三档规模的关键数字、局限。
- `raw/deepseekmoe-2401-06066.pdf` — 33 页 PDF 归档（`*.pdf` 被 gitignore 排除）。

## 修改

- `wiki/index.md` — Sources 段插入 `source-dai-2024-deepseekmoe`（字母序位于 `source-dag` 与 `source-dao` 之间），Techniques 段插入 `deepseekmoe`（位于 `deep-model-fusion` 与 `deep-state-space-model` 之间）。
- `wiki/log.md` — 追加一条 ingest 记录。
- `wiki/source-deepseek-v2-2405-04434.md` — 「FFN 侧沿用 DeepSeekMoE」后加「机制见 [[deepseekmoe]]」，不展开 V2 配置。
- `wiki/gshard.md` — 谱系句补「后继的 [[deepseekmoe]] 保留 top-$K$ 路由与辅助负载均衡损失的框架，但把每个专家切细并隔离出恒激活的共享专家」，source_count 1→2。
- `wiki/mixture-of-experts.md` — GShard 节末补一句指向 [[deepseekmoe]]（top-$K$ 框架下的结构性改动），source_count 17→18。

## 新建交叉链接

- [[deepseekmoe]] ↔ [[gshard]]
- [[deepseekmoe]] → [[sparse-gated-moe-layer]]（后者未加反向指针）
- [[deepseekmoe]] ↔ [[source-dai-2024-deepseekmoe]]
- [[source-deepseek-v2-2405-04434]] → [[deepseekmoe]]

## 数值备注

- 图 3–6 为矢量图，柱高、坐标刻度、以及图例中 "1"/"2"/"." 一类字符在 `pdftotext` 输出里整字丢失（字体编码偏移 +0x1D），只能取正文结论句；正文亦未给图 3 的归一化柱值、图 4 的屏蔽比例、图 5 的 Pile loss 曲线值。
- 附录 B 的实际标题是 "Comparing DeepSeekMoE with Larger Models"（表 8–10），不是专家分布/冗余分析；冗余分析在 §4.5 与图 4–6。任务书提到的「附录专家分布图」在 PDF 中不存在。
- 表 3 表注写 40.5% 计算量，§5.2.1 正文写 "about 40%"，两处并存照录；表 4 表注与正文均为 39.6%。
- 全文无 sigmoid gate、无 gate bias、无 gate 打分变体消融（`grep -i sigmoid` 为 0 命中）；gate 一律为对专家质心的 softmax，式 (5)(8)(11) 一致。
- 表 2 缺 `# Total Params` / `# Activated Params` 行（原文如此），页面只录表中存在的行。
- 表 5 的 NaturalQuestions/MMLU/CEval/CMMLU/GSM8K 为 0-shot，与表 3/4 的 5-shot、8-shot 不同，未混用。
- 正文与表格在 2B 消融处只给 Pile loss 1.808 / 1.806 / 1.811 三个值（1/2/4 个共享专家），论文自述 1:3 为 "marginally better"。
- 论文 PDF 为 2024-01 v1 预印本，无 ACL 版面信息；「ACL 2024」著录来自用户，页面标注未在 PDF 内核实。
- 任务书假设的「与 PLEX / Mixtral 对比」在论文中不存在：2B 验证的对照是 Dense、Hash Layer、Switch、GShard（表 1），附录另有 GShard×1.2/×1.5 与 Dense×4/×16（表 8–10），全文无 PLEX、Mixtral 字样。
- 任务书假设的「PPL-only 评测口径」不成立：语言建模只有 Pile 交叉熵/BPB 一项，另附十余项下游准确率 benchmark，评测口径已在页面「边界」如实写明。
- 任务书假设的「scaling curve」在论文中不存在：图 1 是 Open LLM Leaderboard 平均性能—激活参数散点，图 7 是训练过程曲线（for reference），全文未拟合 scaling law。
- `deepseekmoe` 页 `source_count` 记为 3 而非任务书写的 1：机制对照表引用了 [[gshard]] 与 [[sparse-gated-moe-layer]] 两个既有源，按 AGENTS.md 的「正文唯一源文件数」口径计 3。
