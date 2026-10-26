# Ingest 报告：longformer-2004-05150

源：Longformer: The Long-Document Transformer（Beltagy, Peters & Cohan, AI2），arXiv:2004.05150v2（2020-12-02）。PDF 归档 `raw/longformer-2004-05150.pdf`（`*.pdf` 被 .gitignore 排除，不入 git），`pdftotext -layout` 全文通读（正文 §1–§8 + 附录 A–D，至第 17 页）。

## 创建
- wiki/source-beltagy-2020-longformer.md — WHY：单源 300–500 字摘要页，覆盖问题/机制/证据/边界，系列命名沿用 `source-<作者>-<年>-<名>` 惯例（同 18 号 `source-child-2019-sparse-transformer`）。
- wiki/longformer-attention.md — WHY：technique 主页。三种注意力（滑窗/空洞/global）的混合矩阵结构、每层 $w$ 与每头 dilation 的自由度、global 的任务驱动用法与独立投影、$O(nw)$ 打分与 $O(n)$ 混合复杂度；分阶段 aligned 数据流含三项关键推导：滑窗+空洞的连通层数（含本页补充的同余类限制）、global 行/列依赖集合的不对称、chunk 实现显存 $O(n^2)\to O(nw)$ 的计数。

## 修改
- wiki/index.md — Sources 段 `bel` 位与 Techniques 段 `longf` 位各插入一条，字母序。
- wiki/pyraformer.md — Overview「稀疏 Transformer 变体」清单中 Longformer 接 `[[longformer-attention]]`（该页此前仅以纯文本提及，sliding window/dilated 不是其方法），`last_updated` 2026-05-31→2026-10-26。
- wiki/patchstg-sparse-attention-form.md — static structured sparsity 一族的 Longformer 提法接机制页，`last_updated` 2026-08-31→2026-10-26。
- wiki/log.md — 追加一条 ingest 条目。

## 新建交叉链接
- [[longformer-attention]] ↔ [[pyraformer]]、[[patchstg-sparse-attention-form]]（互相指向）
- [[longformer-attention]] → [[sparse-transformer-strided-fixed]]（18 号 Sparse Transformer，worktree 已定名）、[[flash-attention]]（20 号 FlashAttention 前向链接，论文未入库、最终页名未定，Obsidian 中暂为未解析链接）、[[native-sparse-attention]]、[[moba-attention]]、[[logsparse-self-attention]]、[[deepseek-sparse-attention]]
- [[source-beltagy-2020-longformer]] ↔ [[longformer-attention]]（source-summary 与机制页互链）

## 数值备注
- 任务提示的 text90 不存在：论文字符级数据集为 text8 与 enwik8（各 100M 字符，90M/5M/5M 切分，附录 B），页面按 text8/enwik8 记录。
- 任务提示的 WVQA 对应论文中的 WikiHop 消融（Table 10：seqlen/global attention/独立投影/MLM 预训练/训练时长）；窗口与 dilation 配置消融在 Table 4（150K 步、phase-1 配置、text8 dev，论文自注该口径的排序未必等于训练到底的排序，其脚注 4）。
- Table 12 的 Phase 5 LR 经 pdftotext 丢失前缀，文本为 `000015625`；按 §4.2「每 phase 学习率减半」正文口径读作 0.000015625（$0.00025/2^4$）。Phase 5 窗口的顶层值在文本转换中整体缺失（Phase 1 的 32–8,192 可读），页面未引用该值。
- Fig 1（运行时间/显存对比）与 Fig 3（LED 输入长度消融）为曲线图：Fig 1 只引趋势（线性显存 vs 全注意力 OOM），Fig 3 只引端点值（R-1 35.21→46.23、R-2 11.54→19.62）。
- 排行榜数字（Table 8：WikiHop 81.9、TriviaQA 77.3、HotpotQA 73.2）为 2020-05 提交时点口径；论文自注 BigBird 随后以 16 倍预训练算力刷新并列为混杂因素（其脚注 9），页面照录两说。
- venue：PDF 内仅 arXiv 标头（v2, 2020-12-02），TACL 出版信息未在 PDF 内核实，页面只写 arXiv v2。
