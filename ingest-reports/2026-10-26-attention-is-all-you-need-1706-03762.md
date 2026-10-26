# Ingest 报告：attention-is-all-you-need-1706-03762

## 源与阅读范围

- 官方源：https://arxiv.org/abs/1706.03762（arXiv:1706.03762v7，cs.CL，2023-08-02 版；NIPS 2017）。
- 全文来源：PDF 归档 `raw/attention-is-all-you-need-1706-03762.pdf`（`*.pdf` 被 `.gitignore` 排除，故只保留本地副本），`pdftotext -layout` 得到 801 行文本在 `.tmp/` 中阅读；未改动任何既有 `raw/` 文件。
- 阅读范围：正文 §1–§7 全文 + 脚注 4/5 + Table 1–4 + 附录注意力可视化说明，逐段读完；Table 3 的列位用 `pdftotext -layout -f 9 -l 9` 复核过一次。

## 创建

- wiki/source-attention-is-all-you-need-1706-03762.md — WHY：本源摘要页，覆盖问题（串行计算不可并行、卷积路径随距离增长）、机制（6+6 层、多头、post-LN 残差、正弦位置、三种注意力用法）、证据（Table 2/3 的 BLEU 与消融）与边界（未实验的受限注意力与外推）。
- wiki/transformer-architecture.md — WHY：仓库此前没有原始 Transformer 的机制页，多头/自注意力/残差只在各自的下游页里零散出现；此页承载编码器-解码器数据流（Stage 0–8 分阶段维度块）、$1/\sqrt{d_k}$ 方差条件、Table 1 复杂度与 Table 2/3/4 数值，供下游页统一回指。
- wiki/sinusoidal-positional-encoding.md — WHY：位置编码相关页（APE、PIPE 变频率、RoPE、ALiBi）都拿「标准正弦编码」当基线但无处可指；此页给出公式、波长几何级数、固定位移的二维旋转推导与 Table 3 行 (E) 对照。

## 修改

- wiki/scaling-factor-sqrt-dk.md — WHY：原页只有博客来源；补「原始出处（§3.2.1）」小节，写入论文脚注 4 的前提与方差结论、加法 vs 点积的对照、$h=8$/$d_k=d_v=64$ 默认尺度，并链接机制页（source_count 1→2）。
- wiki/variant-frequency-positional-encoding.md — WHY：动机段把「标准正弦编码」作为对照，补原文引用与机制页链接，避免基线无出处（1→2）。
- wiki/interpretable-multi-head-attention.md — WHY：TFT 的可解释多头是对标准多头的改造，补标准形式的投影/拼接与 base 配置出处（1→2）。
- wiki/inverted-transformer-architecture.md — WHY：该页声称「不修改任何原生组件」，补原生 post-LN 残差 + FFN 的定义出处（1→2）。
- wiki/probsparse-self-attention.md — WHY：正文以 canonical self-attention 为复杂度基线，补 $O(n^2 d)$ 的原始出处（1→2）。
- wiki/index.md — WHY：按 type 插入 3 条（Sources/Sinusoidal/Transformer），段内字母序，与既有条目格式一致。

## 新建交叉链接

- [[transformer-architecture]] ↔ [[sinusoidal-positional-encoding]]（位置注入在数据流中的位置）
- [[transformer-architecture]] ↔ [[scaling-factor-sqrt-dk]]（缩放推导与失效模式）
- [[transformer-architecture]] ↔ [[interpretable-multi-head-attention]]、[[inverted-transformer-architecture]]、[[probsparse-self-attention]]、[[logsparse-self-attention]]（下游改造与稀疏化）
- [[sinusoidal-positional-encoding]] ↔ [[variant-frequency-positional-encoding]]、[[learnable-patch-position-encoding]]、[[adaptive-positional-encoding]]、[[generalized-positional-encoding-framework]]、[[alibi]]、[[rope]]

## 关键公式与数字的原文位置

| 内容 | 原文位置 |
|---|---|
| $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^{\top}/\sqrt{d_k})V$ | §3.2.1 式 (1) |
| 点积方差 $=d_k$ 的前提与结论 | §3.2.1 脚注 4 |
| 多头拼接与 $h=8$、$d_k=d_v=64$ | §3.2.2 |
| 三种注意力用法（编码器自注意力、解码器掩码自注意力、cross-attention） | §3.2.3 |
| $\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$、$d_{ff}=2048$ | §3.3 式 (2) |
| 正弦编码公式、波长 $2\pi\to10000\cdot2\pi$、平移线性性 | §3.5 |
| 每层 $O(n^2 d)$、串行 $O(1)$、路径 $O(1)$ | §4 Table 1 |
| 学习率 $\text{lrate}=d_{\text{model}}^{-0.5}\min(\text{step}^{-0.5},\text{step}\cdot\text{warmup}^{-1.5})$、warmup 4000 | §5.3 式 (3) |
| base 27.3/38.1（$3.3\times10^{18}$）、big 28.4/41.8（$2.3\times10^{19}$） | §6.1 Table 2 |
| $h=1$ 24.9、$h=32$ 25.4、$P_{drop}=0$ 24.6、$\epsilon_{ls}=0$ 25.3、学习式位置嵌入 25.7 | §6.2 Table 3 |
| WSJ 91.3 / 半监督 92.7 F1，RNN Grammar 93.3 | §6.3 Table 4 |

## 数值备注

- EN-FR big 模型 BLEU：§6.1 正文写 41.0，Table 2 与摘要写 41.8；两处并列，页面与摘要页均未择一。
- Table 3 行 (B)：表内只印出一列 16/32，版面位置落在 $d_v$ 列，正文（§6.2）称该行变化的是 $d_k$，并据此讨论「减小 key 维伤质量」；列归属在版面上不唯一，机制页只引行 (A) 的 $h$ 变体，未用 (B) 支持 $d_k$ 结论。
- Table 3 行 (C) 的 $d_{\text{model}}=256$ 行未在机制页列出（仅列 1024 行），因该行同时改动 $d_k/d_v$ 与 $d_{ff}$，单变量解读不成立；数值 24.5（28M）仍保留在原文，未弃用。
- Table 2 中 ByteNet 只给 EN-DE、Deep-Att + PosUnk 只给 EN-FR，属原文缺列，未补。
- 训练 FLOPs 是「训练时间 × 卡数 × 单卡持续算力」的估计（§6.1 脚注 5），非实测；机制页沿用「训练 FLOPs」措辞。
- 摘要在描述 EN-DE 时写 "28.4 BLEU … by over 2 BLEU"，与 §6.1 "more than 2.0" 一致（28.4 − 26.36 = 2.04）；无冲突。
