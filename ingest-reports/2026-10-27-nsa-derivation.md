# 推导核验报告：Native Sparse Attention（arXiv:2502.11089）

源：`raw/2502.11089-nsa.txt`（25 页文本层）。本轮为优化推导：对照原文逐条复算 [[native-sparse-attention]] 的 Stage 1–5、$N_t$ 推导、门控口径与压缩支路细节，并同步核验 [[source-2502-11089-nsa]]；只补强与修正，不重写页面。

## 修改
- wiki/native-sparse-attention.md — WHY：Stage 1 value 侧输入维原写 $[d_v,l]$（误把窗长当序列长），改为 $[d_v,t]$；Stage 2/3 注解块数统一为 $\lfloor(t-l)/d\rfloor+1$，与页内「按 $+1$ 计」口径自洽；$\psi$「另一组 MLP」改为 [INFERENCE]（原文只称 analogous formulation，是否共享 $\varphi$ 未说明）；补 $N_t$ 推导、两支路门控口径算例、式(9) 适用前提与下标重叠权重、门控 MLP 结构未给的标注。
- wiki/source-2502-11089-nsa.md — WHY：解码读取量句子区分 §5.2 文字口径与 Table 4 的 $\lfloor s/d\rfloor+nl'+w$ 取整口径，避免与概念页新推导冲突；last_updated 刷新。

## 核验结论（通过项）
- 式 (3)–(6)：重映射框架、$g^c_t$ 由 MLP 与 sigmoid 得到、$N_t=\sum_c\operatorname{size}[\tilde K^c_t]$，誊写与归因无误。
- 式 (7)：索引集 $0\le i\le\lfloor(t-l)/d\rfloor$、$\varphi$ 为带 intra-block position encoding 的可学习 MLP、$d<l$ 减轻信息碎片化，与 §3.3.1 一致。
- 式 (8)–(12)：重要性分复用压缩分数、适用前提 $l\le l'$、$d\mid l$、$d\mid l'$、GQA 组内求和、降序 rank top-$n$、Cat 拼接得 $nl'$，与 §3.3.2 一致。
- 超参：$l=32,d=16,l'=64,n=16$（含固定激活 1 首块 + 2 局部块）、$w=512$（§4.1）；$g=4$、$h=16$、$d_q=d_k=192$、$d_v=128$（§4.1、§5）。
- Table 1（10 列）、Table 2（6 行 × 11 列）、Table 3（4 格）、Table 4（4 档）逐格与文本层数值一致；§4.3 的差值（DROP +0.042、HPQ +0.087、LCC +0.069 等）复算无误；2560 预算与 leading 128 + local 512 表述一致。
- §3.4 kernel：压缩/窗口兼容 FlashAttention-2、选择支路专用、组中心装载 $Q\in\mathbb R^{[h,d_k]}$、$K\in\mathbb R^{[B_k,d_k]}$、$V\in\mathbb R^{[B_k,d_v]}$ 且 $B_k\mid l'$、外层交给 grid 调度器，逐条相符。
- Figure 6：forward 2.1×/3.8×/6.3×/9.0×、backward 1.1×/2.0×/3.4×/6.0×，与文本层一致；backward 面板的孤立「9.0 ×」PGF 残留维持按正文取值。
- 训练：270B（§4.1）对 260B（摘要、§1）两处口径维持并列；§6.1 的 3B 替代方案对照表述无误；AIME 采样设置（温度 0.7、top-$p$ 0.95、16 条取均值）无误。
- frontmatter：两页 confidence 均已是 medium、status active，无 severe 项。

## 修正项
1. Stage 1 value 侧输入维 $[d_v,l]\to[d_v,t]$：value 侧是 analogous formulation，输入同为整段前序序列，窗长 $l$ 是滑窗参数不是输入维。
2. Stage 2/3 注解块数 $\lfloor(t-l)/d\rfloor\to\lfloor(t-l)/d\rfloor+1$：与式 (8) 原文长度及页内「按 $+1$ 计」声明一致。
3. $\psi$ 是否与 $\varphi$ 共享：原文未写明，原「$\psi$ 为另一组 MLP」降为 [INFERENCE] 并注明 analogous、未给记号。
4. Stage 1 K 侧注解的 $\bcancel{t}$ 移除：压缩是滑窗池化不是矩阵乘消维，统一 $\xrightarrow$ 口径。

## 新增推导
- $N_t=\lfloor(s-l)/d\rfloor+1+nl'+w$ 及 64k 三口径：5630（§5.2 文字）/ 5631（式 7 索引集）/ 5632（Table 4）；差 2 定位为 $l/d=2$ 整除下 $\lfloor s/d\rfloor=\lfloor(s-l)/d\rfloor+2$ 的取整口径差；Table 4 四档（2048/2560/3584/5632）全部按 $\lfloor s/d\rfloor+nl'+w$ 复现，$nl'+w=1536$ 三处一致。
- 门控两口径算例（[INFERENCE]）：两支路各 2 key、分数 $(2,0)$，支路内权重 $0.881/0.119$（分母 $8.39$），拼接后 $0.440/0.060$（分母 $16.78$），权重精确减半；门控全开时权重和为支路数 2，拼接口径恒为 1，故两者不存在重参数化关系。
- 式 (9) 下标重叠（[INFERENCE]，式面直算）：默认超参下 8 项落 $4j-4..4j$ 共 5 个不同下标，权重 $1,2,2,2,1$。

## 数值备注
- 式 (7) 自身两处口径：索引集给出 $\lfloor(t-l)/d\rfloor+1$ 个块，紧随的声明却写 $\tilde K^{cmp}_t\in\mathbb R^{d_k\times\lfloor(t-l)/d\rfloor}$；式 (8) 按 $+1$。属论文内部不一致，页面按式 (8) 取 $+1$ 并保持注明。
- 式 (8) 未写 $1/\sqrt{d_k}$ 缩放，视为继承式 (2) 约定，页面照原文誊写未补。
- §5.2 写「at most $\lfloor(s-l)/d\rfloor$ compression tokens」，at most 吸收了式 (7) 索引集的边界块 $+1$；三口径关系如上。
- 门控的输入特征与 MLP 结构原文未给（只说 input features via an MLP and sigmoid），页面保持示意记号并注明。
- raw/ 本轮只读，未改动；主仓库 /home/jcheng/Documents/llm-wiki 经 `git status` 确认无脏改动。
