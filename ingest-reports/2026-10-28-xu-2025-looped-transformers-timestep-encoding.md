# Ingest 报告：On Expressive Power of Looped Transformers（Xu & Sato, ICML 2025）

源文件：`raw/xu-2025-looped-transformers-timestep-encoding.pdf`（用户提供路径 `C:\Users\HP\Downloads\xu25x.pdf`，从 `/mnt/c` 拷贝入库，原件未改动）

## 创建

- `wiki/source-xu-2025-looped-transformers-timestep-encoding.md` — WHY：每个 `raw/` 源一份摘要，记录作者/会议/题名、核心论点（Theorem 3.6 的 rate + 两个多出的 modulus + TMLT）、关键数值与自述局限。
- `wiki/looped-transformer-approximation-rate.md` — WHY：既有 wiki 只有 looped 架构的结构页（[[recurrent-depth]]）与 UT 页，没有表达力理论页；三种 modulus、rate 公式、loop 预算与「weight-tied FF 不能逐 step 调步长」这条机制需要独立落点，否则只能塞进 Huginn 页造成两条知识链混在一起。
- `wiki/timestep-modulated-looped-transformer.md` — WHY：TMLT 是可复用的架构技术（timestep 编码生成 α/γ 调制），有完整前向数据流与独立的实验证据表，与理论页分开。

## 修改

- `wiki/recurrent-depth.md` — WHY：新增「表达力：逼近率与 timestep 调制」一节，把该路线的理论结果接进来，并写清 Huginn（拒绝 step embedding）与 TMLT（条件化 scale）不是对同一个量的分歧；source_count 4→5，补脚注。
- `wiki/universal-transformer.md` — WHY：「谱系与后续」补一段理论侧定位：UT 固定 $T$ 步属于被分析的权重绑定族，但该分析限单层、固定长度、排除 LayerNorm，不覆盖多层栈与 ACT；source_count 2→3。
- `wiki/index.md` — WHY：按 type 三段各插入一条（Sources / Concepts / Techniques），段内字母序；刷新 `last_updated`。

## 新建交叉链接

- [[looped-transformer-approximation-rate]] ↔ [[timestep-modulated-looped-transformer]]
- [[looped-transformer-approximation-rate]] ↔ [[recurrent-depth]]
- [[timestep-modulated-looped-transformer]] ↔ [[recurrent-depth]]、[[universal-transformer]]
- [[timestep-modulated-looped-transformer]] ↔ [[rmsnorm]]、[[sinusoidal-positional-encoding]]
- [[source-xu-2025-looped-transformers-timestep-encoding]] ↔ 上述三页

## 数值备注

- **Theorem 3.6 的括号**：PDF 两栏排版里 $(Nd)^{1/p}$ 后无可见括号界定作用范围。由式 13（$(Nd)^{1/p}[\omega^{\mathrm{tok}}+\omega^{\mathrm{cont}}]$）与式 10（$\omega_f$ 项不带该因子）推回，页面按「乘两项 modulus、$\omega_f$ 独立成项」转写，并在页面「口径备注」注明。
- **式 16**：原文排成 $\delta^{-1}\cdot2\delta^{-(N+1)d}\ge r-N$，与式 15 的等式方向不一致；页面只引用式 17 的结果，不复刻该行字面形式。
- **Figure 2 caption**：`1/δ −1` 为压平的 $\delta^{-1}$（同一 PDF 内所有上下标均被压平），读作插值把误差缩小 $\delta$ 倍。
- **$b_5$ 维度**：正文写 $W_5\in\mathbb{R}^{4d\times d}$ 而 $b_5\in\mathbb{R}^{d}$；按 $4d$ 维输出应为 $\mathbb{R}^{4d}$。页面照录原文符号，未改正。
- **batch size 冲突**：附录 C.1.3 写通用 batch 64，C.3 写语言建模 batch 48；两处分别引用，不择一。
- **Step 3 的 loop 数**：正文写 $K-1$ 次 loop，loop 预算处写 $2\delta^{-(N+1)d}-1$；论文未显式写出 $K$ 与 $\delta$ 的关系，页面并列两种口径。
- **缺失统计**：Table 2/3/4 均为单值单元格，无标准差、无种子数、无比参数列；reasoning 任务未给参数量（语言模型只写「按参数量对齐」）。
- **原文拼写照录**：Table 2 caption 结尾双句号、附录 D 标题 `Disccusion`、Figure 5 图例 `time depnedncy`。
