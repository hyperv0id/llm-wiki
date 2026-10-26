# 推导核验报告：Multi-token Prediction（arXiv:2404.19737）

源：arXiv 官方 PDF（`arxiv.org/pdf/2404.19737`，29 页），`pdftotext -layout` 提取至 `<worktree>/.tmp/mtp-paper.txt`（1622 行）。**本 worktree 的 `raw/` 中不存在 Gloeckle PDF**（`raw/` 71 个文件中无 `gloeckle*`/`2404*`，git 历史亦无）；既有 ingest 报告 `gloeckle-2024-multi-token-prediction.md` 末尾记录的 `raw/gloeckle-2024-multi-token-prediction.pdf` 路径在当前分支 `ingest/31-mtp-20261027` 上不成立，本轮以 arXiv PDF 为准核验。raw/ 只读，未改动。

本轮为优化推导：对照原文逐条核验 [[multi-token-prediction]] 与 [[source-gloeckle-2024-multi-token-prediction]]，补数据流与显存推导，不重写页面。

## 修改
- wiki/multi-token-prediction.md — WHY：方法节原只有一行 softmax 公式，无分阶段数据流与显存推导。新增 Stage 1–3 `aligned` 块（trunk → 逐头 → 求和）、显存账小节（含 Table S5 口径 callout）、推理侧 self-speculative 对照小节；「何时有效」表加出处列并把「约 ≥3B 反超」回打到表 S7 逐尺寸数字；机制段三条补 §5/附录 L 落点并改题「论文自述推测」；局限段补引用与具体数字。last_updated 已是 2026-10-27，source_count 3 不变。
- wiki/source-gloeckle-2024-multi-token-prediction.md — WHY：损失记号 $L_n$ 对齐原文 $L^n$（式 (2)）；全部事实句补自引用脚注与关键数字的表/图/节出处；frontmatter source_count 0→1、confidence low→medium、last_updated → 2026-10-27。

## 核验结论（通过项）
- **§2 方法**：损失分解 $L^n=-\sum_t\sum_{i=1}^n \log P_\theta(x_{t+i}\mid z_{t:1})\,P_\theta(z_{t:1}\mid x_{t:1})$、架构 $P_\theta(x_{t+i}\mid x_{t:1})=\mathrm{softmax}(f_u(f_{h_i}(f_s(x_{t:1}))))$、头为 transformer 层、unembedding 共享、公平对比加 n−1 头层减 trunk n−1 层，逐条与原文一致。头间独立是 parallel 架构设定，causal/anticausal 变体才引入头间级联（Appendix B）。
- **§2 显存**：朴素实现 logits 及梯度「both of shape $(n,V)$」同时物化；逐头串行 forward/backward、logits 用后即释放、长期只存 $d$ 维 $\partial L^n/\partial f_s$；峰值 $O(nV+d)\to O(V+d)$；「at no expense in runtime (Table S5)」。replicated unembedding 需 $(d,nV)$ 矩阵、不可行（Appendix B）；byte 模型恰好采用 replicated unembeddings（Table S13 图注）。
- **§3.1 规模**：300M–13B 六档、≥91B code tokens；摘要 13B 多解 12% HumanEval / 17% MBPP；表 S7 复核：0.3B（1.0 对 1.8）与 0.6B（3.0 对 4.7）n=4 全面更差，1.3B 混合，3B 起 n=4 六项首次全面不低于 n=1，13B MBPP@1 30.5 对 26.0（相对 +17.3%）、HumanEval@1 15.8 对 14.1（+12.1%），与摘要相对口径自洽。
- **§3.2 加速**：greedy self-speculative（Stern et al. 2018）、xFormers、7B 4-token；code 3.0×、平均接受 2.5/3；表 S2 精确值 code 3.05×（3.50 tokens/forward）、wikipedia 2.74×、books 2.67×；上界 $k$（表 S2 图注）；batch 1–40 稳定（图 S10）。
- **§3.3 byte**：7B、314B bytes ≈ 116B tokens、8-byte 预测 MBPP@1 +67%、HumanEval@1 +20%；6.4×（表 S3 精确值 6.39×，batch 16）；「1.7× less data」表述一致。
- **§3.4 最优 n**：n=4 在 HumanEval/MBPP 全 pass@k 领先（+3.8/+2.1/+3.2 与 +1.2/+3.7/+4.1），APPS/Intro n=6 领先（+0.7/+3.0/+5.3）；字节级 n=8 更一致；温度为 oracle（表 S12）。
- **§3.5–3.7**：1T（4 epoch）MBPP@1 +2.4%、HumanEval@100 +3.2%；CodeContests 微调 n=4,n′=1 最佳（图 4、附录 F 温度 oracle 口径）；NL 基准 n=2 持平、n=4 略退（图 5）；摘要 ROUGE-L 平均增量 200B +0.51/+0.46 → 500B +0.28/+0.31（表 S9）；GSM8K 200B 时 n=2 好、500B 翻转、n=4 全程差（图 S13）。
- **§4–5 与附录 L**：induction ≤30M 大幅改善、≥100M 优势消失（§4.1）；多项式算术 OOD 提升、30M→100M 约 3 倍扩参不如换损失（§4.2、图 S16）；$\frac{n(n+1)}{2}$ 项与平均 $\frac{n+1}{2}$ 倍（§5.1、L.3）；相对互信息 $I_{p\|q}(X;Y)=D(p\|q_X\otimes q_Y)-D(p\|q)$、权重 2（§5.2、L.2）；scheduled sampling 不适用论证（L.1）；Llama-2 微调无显著改善、归因「changes the initialization too brutally」（表 S6）。
- 运行时 Table S5：n=2 为 1.02–1.07×、n=4 为 1.07–1.22×，归因逐头 backward 失去 FSDP overlap、可重实现消除。

## 修正项
1. 损失记号 $L_n \to L^n$：原文用上标（式 (1) $L^1$、式 (2) $L^n$）。
2. 「约 ≥3B 起代码任务持续反超」原为图 3 目测：图内逐尺寸数字进不了文本层，改以表 S7 逐尺寸数据表述（0.3B/0.6B 全面更差、1.3B 混合、3B 起 n=4 六项首全面不低于 n=1）。
3. 机制段原第 2、3 条无引用，且段题「推测」未标明推测主体；改为「论文自述推测」并逐条落 §5.1/L.3、§5.2/L.2、§4.1–4.2——全部条目均为原文推测或实证，页面不再残留无归属的推断。
4. source-summary frontmatter source_count 0 且正文无脚注，与「每个事实句带引用」规则冲突；补自引用脚注与出处，source_count 0→1、confidence low→medium。

## 新增推导
- **Stage 1–3 数据流**（对齐 SKILL「数据流与维度变换」）：$z_{t:1}=f_s(x_{t:1})\in[T,d]$；$h^{(i)}=f_{h_i}(z_{t:1})$（逐头独立、形状不变）；$\ell^{(i)}=f_u(h^{(i)})$：$[T,\bcancel{d}]\times[\bcancel{d},V]\to[T,V]$（共享 unembedding 消去 $d$）；softmax 沿 $V$ 维；损失对 $i,t$ 求和为标量。条件独立假设写明：式 (2) 把联合 $\log P(x_{t+n:t+1}\mid z)$ 直接展开为 $n$ 项之和，头间自依赖仅存在于 causal/anticausal 变体。
- **显存账**：朴素并行同时物化 $(n,V)$ 的 logits 与梯度 → $O(nV+d)$；串行逐头 forward/backward、即用即释放、trunk 梯度 $d$ 维长期驻留 → $O(V+d)$；对照 $(d,nV)$ replicated unembedding 反例与 byte 模型特例。
- **self-speculative 对照**：Stern 线性头 → transformer 头；训练侧用全部损失项 vs Stern 随机选一头；Medusa top-k 兼容；上界 $k$；表 S2/S3 数字。Qwen3-Next 原生 MTP 头维持既有「工程落地」节的定位区分，未混入。

## 数值备注
- 正文「at no expense in runtime」（§2）与表 S5 的 n=4 1.07–1.22× 并存：论文归因 FSDP overlap 损失、可重实现消除。页面以 callout 并列两口径，不替论文取其一。
- NLP 基准个数论文内部不一致：正文 §3.7 写「6 standard NLP benchmarks」，附录 G 文字列 7 个（ARC Challenge、COPA、Hellaswag、Natural Questions、PIQA、SIQA、TriviaQA），图 S12 图注列 8 个（多 MMLU）。页面不采用具体个数，只写「multiple-choice NL 基准」。
- 摘要「12% HumanEval / 17% MBPP」为相对口径，与表 S7 绝对值（15.8/14.1、30.5/26.0）换算一致；引言的「around 15% more on average」为两者的平均表述。
- 表 S2 code 模型训于 1T tokens、文本模型训于 500B tokens（图注口径）；主文 §3.2 的「2.5 accepted out of 3 suggestions」与表 S2 的 3.50 tokens/forward 相容（1 + 2.5 = 3.5）。
- raw/ 本轮只读，未改动；主仓库 /home/jcheng/Documents/llm-wiki 经 `git status --porcelain` 确认无脏改动；本 worktree 状态干净，未 commit。
