# Ingest 报告：On Layer Normalization in the Transformer Architecture（arXiv:2002.04745v2）

源：`raw/on-layer-normalization-in-the-transformer-architecture-2002-04745.pdf`（17 页；正文 §1–§5，附录 A–H）。通读范围：§1 Introduction、§2 Related work、§3.1 Post-LN 定义、§3.2 warm-up 实验、§3.3 初始化理论（Definition 1、Theorem 1、Lemma 1–3）、§3.4 经验校验、§4 Experiments、§5 Conclusion、附录 A（实验设置）、B–E（Lemma 1/2/3 与 Theorem 1 的证明）、F（其他层与参数的扩展分析）、G（$(\epsilon,\delta)$-bounded 例子）、H（小学习率实验）。数学式以 arXiv e-print LaTeX 源核对，避免 PDF 文本层的分式与根式错位。

## 创建

- `wiki/source-xiong-2020-layernorm-placement.md` — WHY：全库此前只有组件级提及（`convergent-normalization`、`guided-layer-normalization` 等），没有这篇 paper 的落点；源提供了定理、证明与四个任务的实验，够一条完整论断链，符合 source-summary 的建立条件。
- `wiki/pre-ln-vs-post-ln-analysis.md` — WHY：仓里没有 layer-norm 放置的机制页，而「LN 在残差和内外」是 Transformer 训练能否去掉 warmup 的直接解释。页内收敛两种放置的分阶段前向、层间雅可比连乘的推导、定理 1 的完整前提与两条梯度界，以及「哪部分是定理、哪部分是附录的初步分析」的分界。

## 修改

- `wiki/identity-mapping-property.md` — WHY：该页把残差块写成 $x_{l+1}=x_l+F(x_l)$，恰是 Pre-LN 的形式；Post-LN 的 $x_{l+1}=\mathrm{LN}(x_l+F)$ 每层终止恒等路径，这一对照在页内没有。新增「LN 放置对恒等路径的影响」一节并补引用，source_count 1→2、last_updated 刷新。
- `wiki/index.md` — WHY：Sources 段加 `source-xiong-2020-layernorm-placement`（按字母序落在 xcpd 与 yang 之间）、Techniques 段加 `pre-ln-vs-post-ln-analysis`（落在 post-hoc-forecast-revision 与 prediction-refinement 之间），并刷新 last_updated。

## 新建交叉链接

- [[source-xiong-2020-layernorm-placement]] ↔ [[pre-ln-vs-post-ln-analysis]]
- [[identity-mapping-property]] ↔ [[pre-ln-vs-post-ln-analysis]]
- [[pre-ln-vs-post-ln-analysis]] ↔ [[pre-ln-transformer]]（按名链接；该页由 07 号任务另建，本任务不建同名页）

## 数值备注

- **正文与正文自相矛盾（BERT 加速比）**：§4.2 写「Post-LN 在 500k 步达到 1.69 验证损失，Pre-LN 在 700k 步达到同等水平，说明有 40% 的加速」，两个步数的方向与「加速」相反（700k > 500k），而同段又把「加速量」写成 200k。页内两处并列（步数与「40%」称号）并标注方向冲突，未替论文选一个口径。
- **没有学习率上限表达式**：任务预期存在 $\eta\leq\cdots$ 形式的界，但全文（含定理 1 与附录 E 的证明）只给出梯度尺度 $\mathcal{O}(d\sqrt{\ln d})$ 与 $\mathcal{O}(d\sqrt{\ln d/L})$，从未把梯度尺度换算成学习率上限。页内据此只写梯度界，并把「大 lr 发散」明确归为实验观察（附录 H 固定 $10^{-4}$ 可训但更慢、§4.2 Post-LN BERT 用 $3\times10^{-4}$ 直接发散），不补公式。
- **定理编号**：任务记作 Theorem 3.1，arXiv v2 全文编号为 Theorem 1（§3.3 内，正文与附录 E 均引用「Theorem 1」）。页面统一用 Theorem 1，未出现 3.1。
- **终值缺表**：全文只有一张表（Table 1，两种放置的公式对照），BLEU / 验证损失 / 准确率全部以曲线呈现（Figure 2、3、4、5、6）。散文里的数字只有：无 warmup BLEU 8.45、带 warmup 约 34、$T_{\mathrm{warmup}}=500$ 时 31.16 与 2.77、Pre-LN 约 34 与验证损失约 4、脚注 1 的 35.5 test BLEU、BERT 的 1.69。WMT14 En-De 终值、GLUE 的 MRPC/RTE 准确率均只在图中，页面不补数字。
- **「Pre-LN 终值略低」不成立**：论文对 IWSLT 的表述是 comparable，且脚注 1 记录 Pre-LN 可达 35.5 test BLEU（比正文「约 34」更高，但 dropout 0.3 与 batch 8192 的配置也不同）。页面按 comparable 写，未引入「略低」的判断。
- **Table 1 的读法**：PDF 文本层把该表两列交错输出（同一行左右列相邻打印），分式与下标亦断裂；页面按 arXiv e-print 的 LaTeX 源逐行核对，确认 Post-LN 列的残差加在归一化之前（$x^{post,2}=x^{post}_l+x^{post,1}$ 后才 $\mathrm{LN}$）、Pre-LN 列的残差加在未归一化的流上（$x^{pre,3}=x^{pre}_l+x^{pre,2}$），与正文「LN 在残差和内外」的表述一致。
- **附录 F 的范数估计是近似**：$\|\mathbf{J}_{LN}\|_2\approx\sqrt{2/3}$ 依赖 $\|x^{post,5}\|_2^2=\frac{3}{2}d$ 的集中性，论文自己标为「$\approx$」与「preliminary analysis」；页面把它与定理 1 分开叙述。
- **$(\epsilon,\delta)$ 的实测值只有一处**：§3.4 记 6-6 模型的隐藏态满足 $(0.1,0.125)$-bounded，没有跨规模的对应记录。
