# 推导核验报告：DeepSeek Sparse Attention（2026-10-27）

## WHY

本轮是对 [[deepseek-sparse-attention]] 的优化推导：对照原文（`raw/2512.02556-deepseek-v3-2.md`，转自 arXiv HTML v1）逐条复算既有推导并补强，不重写页面。

## 版本与编号核对

- **版本漂移**：arXiv abs/2512.02556 当前 latest HTML 与 v1 的正文（S1 起至参考文献前）剥离页面 chrome 后逐字一致，raw 转换稿没有版本漂移。
- **式编号**：§2 编号公式只有式 (1)–(4)（HTML 锚点 S2.E1–S2.E4）：(1) indexer 打分、(2) 稀疏注意力、(3) warm-up KL、(4) sparse-on-selected KL。式 (5) 起属于 §3.1 的 GRPO 目标。wiki 引用的「式 1」「式 2」编号无误。
- **Figure 3**：v1 HTML 的 S2.F3 元素同样只剩图题「(a) Prefilling」，图为位图无数据标签；§2.3 正文无任何成本或加速比数字。

## 核验结论（逐项）

- **Stage 1**：$Q^I\in[L,H^I d^I]$、$w^I\in[L,H^I]$、$K^I\in[L,d^I]$ 三条投影维度与式 1 的角色分配（$w$、$q$ 由 $\mathbf{h}_t$ 导出，$k$ 无头下标）全部复算通过；汇合块 $[H^I,L,d^I]\times[d^I,L]\to[H^I,L,L]\to[L,L]$ 与式 1 一致。
- **Stage 2**：$[L,L]\to[L,k]$ 沿前序维逐行 Top-$k$，与式 2 及「preceding token」因果限定一致；$k=2048$ 仅见于 §2.1.1 稀疏训练阶段。
- **Stage 3**：latent 投影 $[L,d_c]$、gather $[L,k,d_c]$、打分 $[n_h,k]$、softmax 沿 $k$、输出 $[n_h,d_h]$ 复算通过；投影与 $d_c$ 本就标 [INFERENCE]，范围不变。
- **Stage 4**：拼头 $[1,n_h d_h]\times[n_h d_h,d]\to[1,d]$ 无误。
- **训练流程**：两阶段（warm-up 1000 步冻结非 indexer 参数 / sparse 15000 步全参数）、sparse 选择在 warm-up 之后整体引入、detach 语义（indexer 信号仅 $\mathcal{L}^I$、主模型仅 LM loss）均与 §2.1.1 原文一致。
- **矛盾检查**：未发现 wiki 论断与原文冲突，无历史论断/争议条目需要落。

## 修改

- `wiki/deepseek-sparse-attention.md` — WHY：补五处推导（ReLU 单头非负/总分可负、top-$k$ 的 logits 域排序与上限语义、latent 粒度 gather 的省算理由、省算量不等式链、top-$k$ 梯度路径与 $p_{t,:}$ 口径推断），两处 token 复算，一处 Figure 3 可得性表述精化；frontmatter 去重 `source_count`（3 与 2 并存，取 3：实际引用 3 个源）；`last_updated` → 2026-10-27。
- `wiki/log.md` — 追加一条 maintenance 记录。

## 新建交叉链接

- [[deepseek-sparse-attention]] → [[multi-head-latent-attention]]（正文 latent 首次出现处 + 相关页面列表；MLA 页此前无 DSA 入链）

## 数值备注

- **复算吻合**：warm-up $1000\times16\times131072=2.097\times10^9\approx2.1$B；sparse $15000\times480\times131072=9.437\times10^{11}\approx943.7$B，两处论文数字与步数×序列×128K 一致。
- **§2.2 可复算项**：唯一数字 AA-LCR reasoning 模式 +4 分（无绝对分）；ChatbotArena 仅「closely matched」（2025-11-10 评测），无 Elo 数值；标准基准无表无分。
- **§2.3 可复算项**：无。论文未给 kv cache 大小、时间或加速比数字，Figure 3 无数据标签，本轮不存在成本代入复算项，页面维持「无法引用具体减免幅度」。
- **未给数值的超参**：$H^I$、$d^I$、$d_c$、$n_h$、$d_h$ 原文均无数值，省算量不等式链保持符号形式；masked MHA 模拟 DSA 的短序列阈值未给。
- **口径推断（已标 [INFERENCE]）**：$p_{t,:}$ 的「attention scores」按 KL 目标推断为 softmax 后概率；top-$k$ 无 straight-through 的表述为「论文未给实现细节」而非原文论断。

## Lint 修正（2026-10-27）

独立核查后就地修复三处：「打分的非负性」段的 $p_{t,:}$ 表述按论文原词改回「注意力分数求和」，口径推断显式指向训练节，删去「单头非负使 Softmax 可逐头对齐非负混合」（softmax 输出恒非负，该因果不成立）；$p_{t,:}$ 口径推断中 logits 反例的理由由「L1 归一化后不构成分布」更正为「负分取绝对值后不再与注意力分数成比例」（L1 归一化对任意向量都产出单纯形向量）；两阶段顺序的理由由「绕开选择不可导与无训练信号」更正为「冷启动时 indexer 未初始化、LM 损失因选择不可导给不了梯度，先 dense KL 对齐，引入 top-$k$ 时选择才落在有意义的分数上」。其余推导与数值复算核验通过，无待裁决项。
