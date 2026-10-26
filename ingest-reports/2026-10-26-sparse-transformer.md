# Ingest 报告：Generating Long Sequences with Sparse Transformers (arXiv:1904.10509v1)

## 源文件
- `raw/sparse-transformer-1904-10509.pdf` — 本次入库时下载归档（`*.pdf` 在 `.gitignore`，不入 git）；pdftotext 文本留在 `.tmp/sparse-transformer.txt`。

## 创建
- `wiki/source-child-2019-sparse-transformer.md` — WHY：本源首入库，落问题（$O(n^2)$ 注意力瓶颈）、机制（$p{=}2$ 稀疏分解 + 三项工程改动）、四组数据集证据（Table 1–4）与边界（只测密度建模、strided 文本失配、音频容量换算）。
- `wiki/sparse-transformer-strided-fixed.md` — WHY：strided/fixed 两模式、连通性论证（§4.2 判据 + 两步构造）、pre-activation 残差的跨块计算、$O(n\sqrt{n})$ 复杂度需要一页技术载体；数据流按分阶段 aligned 块写（嵌入 → strided 头 → fixed 头 → 残差/前馈 → 实现差异）。

## 修改
- `wiki/logsparse-self-attention.md` — WHY：任务描述称其为 TODO 空壳，实际是 LogTrans（Li et al. 2019）LogSparse 的完整页面，缺的只是 `index.md` 里那条摘要；**不合并**——它与本文的 strided/fixed 是不同论文的不同机制（指数间隔 vs stride 块/摘要单元），合并会让两个知识链条互相污染。仅把「Relationship to Other Sparse Attention Methods」表中 Sparse Transformer 行接入 [[sparse-transformer-strided-fixed]] 与本源页，`last_updated` 刷新。
- `wiki/probsparse-self-attention.md` — WHY：区分小节的 Sparse Transformer 一条原是无链接散名，接入新页与本源页并补脚注定义（source_count 2→3），避免与 ProbSparse 混同。
- `wiki/index.md` — WHY：Sources、Techniques 段各加一条（字母序：source-cheng 与 source-chronos 之间、sparse-teacher-forcing 与 spatial-imputation-decoder 之间）；顺带把 [[logsparse-self-attention]] 的 `— TODO` 摘要补成真实一句话（页面本体早已完整，属于 index 维护欠账）。
- `wiki/log.md` — WHY：追加一条 ingest 记录。

## 新建交叉链接
- [[sparse-transformer-strided-fixed]] ↔ [[logsparse-self-attention]]
- [[sparse-transformer-strided-fixed]] ↔ [[probsparse-self-attention]]
- [[sparse-transformer-strided-fixed]] ↔ [[transformer-architecture]]
- [[source-child-2019-sparse-transformer]] ↔ [[sparse-transformer-strided-fixed]]
- [[source-child-2019-sparse-transformer]] → [[fast-weight-mlp-view-of-attention]]（仅在相关页面列表标注「本文未使用该术语」，防混同）

## 数值备注
- **任务清单所列实验与论文不符**：text8、BooksCorpus、Wikipedia、CNN/DailyMail、「天体摄影」均不在这篇论文中。§7 的四组实验是 CIFAR-10、Enwik8、ImageNet 64×64、12kHz 古典音乐音频；Wikipedia 仅作为 Enwik8 的语料来源描述出现（§7.2）。全文无 BLEU，无机器翻译/摘要任务。页面按论文实际实验写，未补写清单条目。
- **「fast weights（跨块缓存 K/V）为何不增加参数」**：论文没有 fast weights 机制（全文无该词，grep 证实）。参数不随 $n_h$ 变的论证在 §5.1——attend 内部权重矩阵维度按 $1/n_h$ 收缩。技术页如实写这一点，并加一句防混同说明。
- **「strided 连通的 $\lceil\cdot\rceil$ 论证」**：论文正文只有 §4.2 的路径判据（路径至多 $p+1$ 个位置）与 §4.3 的集合定义；每头依赖集大小（strided $l{+}1$ 与 $\lfloor i/l\rfloor{+}1$、fixed 约 $l$ 与 $(c{+}1)(\lfloor i/l\rfloor{+}1)$）与两步构造 $a=i-l\lfloor(i-j)/l\rfloor$ 是技术页推导，标 [INFERENCE] 并写出前提（$i\ge l$ 时 $l{+}1$ 个连续窗口位置对每个模 $l$ 剩余类至少含一个代表；lint 核查修正——初稿的 $a=i-((i-j)\bmod l)$ 在 $i-j\ge 2l$ 时 $j$ 落在 $a$ 的窗口外，不满足 §4.2 判据）。
- **pdftotext 丢根号**：§4.3 "chosen to be close to √n" 的 √、§7.4 "4√4 = 8" 的根号在文本层丢失，按上下文与 Table 4 的参数比例（152M→25M→3M，恰为 ×4 长度 ÷8 容量）恢复；位置嵌入初始化 $0.125/\sqrt{d\,n_{emb}}$ 由层叠分式复原。
- fixed 模式摘要单元按原文集合 $\{l-c,\dots,l\}$ 是 $c{+}1$ 个（含块边界），论文例子 $l{=}128$、$c{=}8$ 对应 120–128 共 9 个位置；「比 strided 约多 $c$ 倍计算」按论文原话照录（§4.3）。
- ImageNet-64 的 3.44 为单次运行（论文标注 "3.437 across 1 run"）；CIFAR-10 与 Enwik8 给了多种子均值（$\pm0.004$、$\pm0.001$）。无表格与正文数字冲突；v1 共 10 页，无附录。
