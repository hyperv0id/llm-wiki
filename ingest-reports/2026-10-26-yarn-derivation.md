# Ingest 报告：YaRN（arXiv:2309.00071v3，补推导与数据流）

- **源**：`raw/yarn.pdf` — YaRN: Efficient Context Window Extension of Large Language Models（Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole；Nous Research / EleutherAI / University of Geneva）。PDF 元数据 `arXiv:2309.00071v3 [cs.CL] 6 Feb 2026`，20 页，v5 版在 arxiv.org 直连被限流、`www.arxiv.org/pdf/2309.00071` 可取。
- **通读范围**：正文 §1–§6（含 §3.1 NTK-aware、§3.2 NTK-by-parts、§3.3 YaRN、§3.4 Dynamic Scaling、§4.1–§4.5 实验）＋附录 A.1 PI、A.2 NTK-aware 的 base 变更推导、A.3 温度扫描、B.1–B.7（Table 5–10、Figure 2–8）。`pdftotext -layout` 后逐节核对表号与数字。
- **命名偏离**：批次给的路径是 `raw/yarn-2309-00071.pdf`。按 AGENTS.md 的 slug 规则（slug = raw 文件名去扩展名，与 source-summary 一一对应），该名会产出 `src-yarn-2309-00071`，与仓库现有 6 个页面里的 `[^src-yarn]` / `[[source-yarn]]` 不符，故落为 `raw/yarn.pdf`（`*.pdf` 在 `.gitignore` 内，仅本地副本）。

## 创建

- 无。

## 修改

- `wiki/yarn.md` — WHY：原文只给了 Definition 2 的组合式，旧页既无数据流也无公式。本轮补「数据流与维度变换」Stage 1–5（波长→圈数→三段→分段缩放→温度并进分母→$s$ 的来源），每阶段给维度与物理含义；补「NTK-by-parts 为什么成立」的相位覆盖论证与「工程规则、非定理」标注；补温度缩放的拟合来源（Eq. 15 是扫点拟合，熵与 $t$ 无定量关系）；证据段按 Table 1/3/4/5/6/7/8/9/10 与 §4.1/§4.4 重写；「方法演进谱系」压成组合关系 + 四个组件页链接；旧的 ALiBi 对照表（"~3L / 32L"）在原文无出处，删除。
- `wiki/source-yarn.md` — WHY：`source_count` 1、`confidence: medium` 与正文一致，保留；修正两处口径——「平均 <0.5%」实为 $s=16$ 与 $s=32$ 之间的 0.49%（§4.4），不是与环境基线的差；「首次完整记录」改为 §1 原话「完整记录」。局限性改成论文可核实的边界（Eq. 15 拟合范围、评测止于 131072）。
- `wiki/ntk-aware-interpolation.md` — WHY：旧页「高频少缩放：高维度（大 $d$）…」把 $d$ ↔ 频率标反了（$d$ 小才是短波长高频）。改为「推导」节：$\theta_d'/\theta_d=s^{-2d/(|D|-2)}$，$d=0$ 不动、$d=|D|/2-1$ 恰为 $1/s$，反解即 Eq. 22；并写明它是「高频维继续外推」的非纯插值方案。
- `wiki/ntk-by-parts-interpolation.md` — WHY：删除「无越界外推：所有维度严格在插值范围内」这一与 §3.2 相反的论断（$r>\beta$ 段按设计不插值）；优势条改为可核实表述；加相位论证与边界标定的指向。
- `wiki/attention-temperature-scaling.md` — WHY：补 Eq. 14 的分母 $t\sqrt{|D|}$ 与 $c=\sqrt{1/t}$ 的对应（$q,k$ 各乘 $c$，内积放大 $1/t$）；写明 Eq. 15 的拟合对象（LLaMA 7b/13b/33b/65b，不微调，取 perplexity 最低的 $\sqrt{1/t}$）；标题改为不重复 yarn 页的写法。
- `wiki/dynamic-scaling.md` — WHY：指向 yarn 页 Stage 5，说明它只决定 $s$ 与 $\cos/\sin$ 表的重建时机，不改插值公式。
- `wiki/index.md` — WHY：`[[yarn]]` 条目摘要随内容变化更新（原文 "efficient context window extension" 不含方法构成）。
- `wiki/log.md` — WHY：追加本次 ingest 条目。

## 新建交叉链接

- [[yarn]] ↔ [[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]]、[[attention-temperature-scaling]]、[[dynamic-scaling]]：四页均加回指（Stage 3/4、Stage 5、相位论证），yarn 页只留组合与数据流。
- [[yarn]] ↔ [[alibi]]：ALiBi 段落改引 alibi 页做机制说明，只把 §1 的「有限外推」判断留在 yarn 页。

## 数据流可执行性核对

$|D|=128$、$b=10000$、$L=4096$、$\alpha=1$、$\beta=32$ 代入 Stage 1–2：

- $r(20)=36.66>32$，$r(21)=31.75\le32$ → $\mathcal{D}_{\mathrm{keep}}=\{d\le20\}$，21 个复维，$\lambda_d\le112$。
- $r(45)=1.004\ge1$，$r(46)=0.87<1$ → $\mathcal{D}_{\mathrm{interp}}=\{d\ge46\}$，18 个复维，$\lambda_{63}\approx5.44\times10^4$。
- 其余 25 个复维落在 ramp；$d=30$ 处 $\gamma=0.248$、$s=16$ 时频率 $0.295\theta_{30}$；$d=40$ 处 $\gamma=0.034$、$0.094\theta_{40}$。
- $L=2048$ 时两个阈值各下移约 4.8 个复维（$20.94\to16.12$、$45.03\to40.21$），三段仍非空。

## 数值备注

- **Table 5 的列**：实际是 2048/4096/8192/16384/32768 五列，旧页表只取 2k/8k/16k/32k 四列（跳过 4k），数字无误但列不完全；lint 轮把 4k 列补回（PI 4.95、NTK-aware 3.92、NTK-by-parts 3.75、YaRN 3.77），表与正文引用的 4k 列一致。
- **YaRN 并非每列最优**：Table 5 微调行 YaRN 在 2k（4.19 对 4.14）与 4k（3.77 对 3.75）落后 NTK-by-parts，原文未解释；旧页「唯一单调下降」的说法不成立（NTK-by-parts 五列同样单调），已改。
- **Eq. 21 的指数与 Eq. 8 的 $d$ 索引差一个因子 2**：原文写 $b'^{(|D|-2)/|D|}=s\,b^{(|D|-2)/|D|}$，并称 "the last dimension $d\in D$ is $|D|-2$"；若按 Eq. 8 的 $\lambda_d\propto b^{2d/|D|}$ 且 $d=|D|-2$，指数应为 $2(|D|-2)/|D|$。按 $d=|D|/2-1$（最后一个复维）读才自洽，Eq. 22 的结果与压缩比 $s^{-2d/(|D|-2)}$ 一致。两页均用压缩比表述；lint 轮在 `ntk-aware-interpolation.md` 补一句 $2(|D|/2-1)/|D|=(|D|-2)/|D|$，即两种记法解出同一个 $k$。
- **$r(d)$ 的方向**：批次说明里写作 $r(d)=\lambda_d/L$，原文 Eq. 10 是 $r(d)=L/\lambda_d$（预训练窗口内的圈数）。按原文写：只有 $r>\beta$ 才是短波长高频不插值，若取倒数，$\alpha,\beta$ 的方向会整体反过来。
- **$|D|=128$ 是推断**：论文只写 $|D|$ 为 RoPE 隐藏神经元集合，未给 LLaMA 的具体头维；128 取自 LLaMA 系配置，示例处标 [INFERENCE]（`yarn.md` Stage 2 代入句）。
- **温度与熵**：§3.3 只说「熵上升的性质与 $t$ 有某种普遍性」，未给熵的度量或熵-$t$ 关系；Eq. 15 由 §3.3 的扫点拟合（A.3 是 $s=8$ 的单独扫描，Figure 4–6，896 篇 16k RedPajama 文档）。
- **Figure 2/3 分工**：Figure 2 是训练损失曲线，Figure 3 是 perplexity + passkey；旧页未引，本轮在证据段标清。
- **Eq. 14 的分母**：本页初稿把 Eq. 14 写成 $\sqrt{t|D|}$，与 $c=\sqrt{1/t}$ 的合并式对不上（差一个 $\sqrt{t}$）。按 PDF 该式字形位置（`pdftotext -bbox`：`t` 在根号左侧、`|D|` 在根号下；400 dpi 裁图为 $\frac{q_m^Tk_n}{t\sqrt{|D|}}$）与正文「$q,k$ 同乘 $\sqrt{1/t}$」的 length-scaling 技巧，分母是 $t\sqrt{|D|}$。两页已按此改，`c^{2}=1/t$ 的等式成立。
- **Table 5 的单调性**：微调行 PI 为 5.70/4.95/4.64/3.97/3.57，同样单调下降，只有 NTK-aware 在 16k 之后（32k 列 8.49）上翘；初稿「PI 与 NTK-aware 在高列上翘」把 Dynamic 行的 PI 行为套到了微调行，已改。
- **source-yarn 的标准基准口径**：初稿写「变动在 $\pm1.3$ 以内」，与 Table 3 不符——MMLU 为 $-1.3/-2.1/-3.0/-3.9$（7B $s=16$、7B $s=32$、13B $s=16$、13B $s=32$），其余三项在 $\pm1.7$ 内，已按表改。
- **开头句与 Table 7 列名**：初稿「$s=16$ 时 400 步，即把上下文从 4k 推到 128k」把两阶段并成一步（Table 1：$s=16$ 400 步到 64k，$400+200$ 步的 $s=32$ 到 128k），已按 §4.1 拆开；Table 7 列名按表头写 98304（初稿记作 98k）。
- **PDF 文本层**：`pdftotext -layout` 后 Table 5–10 逐格可读，未发现缺列；公式区（Eq. 3–5、8–14、21–24）编号连续，Unicode 数学符号已转 LaTeX。
