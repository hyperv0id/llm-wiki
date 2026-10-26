# Ingest 报告：kimi-linear-2510-26692

## 创建
- wiki/source-moonshot-2025-kimi-linear.md — WHY：每个 raw/ 文件一份 source-summary，覆盖核心论点（三 regime 超 full attention）、机制（细粒度门 + DPLR 特化 + 3:1 混合）、证据与边界
- wiki/kimi-delta-attention.md — WHY：本源的核心技术贡献，模型结构页按分阶段 aligned 块写前向数据流（Stage 1–6），落全部关键数字（Tables 1/3/4/5/8/9、Figures 1/2/4/5/6/7、式 13–14）
- wiki/gated-delta-rule.md — WHY：任务指定的交叉链接目标 [[gated-delta-rule]] 在库内不存在；KDA 是其细粒度扩展、谱系最近，本源 §2.2/§6.1/Table 6/7 提供了完整的 GDN 定义、TTT 目标与位置编码视角，单源成页（confidence: medium）

## 修改
- wiki/multi-head-latent-attention.md — WHY：补 MLA 在 Kimi Linear 3:1 混合中的用法（NoPE、可转 MQA、KV cache 账），source_count 1→2
- wiki/lightning-attention.md — WHY：补与 KDA 的分块策略同构性及 3:1 对 7:1 的混合比例对照，source_count 2→3
- wiki/minimax-sparse-attention.md — WHY：补论文 §7.1 的线性 vs 稀疏路线对照端点，source_count 6→7
- wiki/mamba.md — WHY：相关页面补 KDA 反向链接与合成任务中 Mamba2 全败的证据，source_count 4→5
- wiki/selective-ssm.md — WHY：「与其他页面」补 KDA 把 Mamba2 更新归入无修正项一类的 Table 7 对照，source_count 2→3
- wiki/index.md — WHY：Sources/Techniques 各按字母序落三条新页面条目
- wiki/log.md — WHY：追加一条 ingest 记录

## 新建交叉链接
- [[kimi-delta-attention]] ↔ [[gated-delta-rule]]（细粒度扩展关系，谱系最近）
- [[kimi-delta-attention]] ↔ [[multi-head-latent-attention]]（3:1 混合的 MLA 支路 + NoPE）
- [[kimi-delta-attention]] ↔ [[mamba]] / [[selective-ssm]]（Mamba2 标量衰减无修正项的对照）
- [[kimi-delta-attention]] ↔ [[lightning-attention]]（同为块内并行/块间递推，3:1 对 7:1 混合比例）
- [[kimi-delta-attention]] ↔ [[minimax-sparse-attention]]（线性 vs 稀疏两条路线的对照）

## 数值备注
- 任务提示中的「大海捞针 / VD 编码」专项评测在本论文中不存在：全文无 needle/NIAH/video 相关内容（grep 复核），长上下文只报 RULER/MRCR/HELMET-ICL/LongBench V2/Frames/RepoQA/Long Code Arena 的聚合分（Table 5，128k），RULER 内部虽含 NIAH 类子任务但论文未拆分报告
- Figure 6（RL 曲线）、Figure 1/7（吞吐/TPOT）图内数值只有坐标轴刻度与个别标注（84.3/81.3/51.0/47.9/47.2、3.98×、6.3×、1.84ms/11.48ms、2.3×/2.9×/2.2×/1.8×）进入文本层；「2.3× at 1M decode」出自正文 §6.3 引 Figure 7b，与 Figure 7b 图内标注 2.2× 不一致，两处并列保留、以正文口径为主
- 附录 D Table 9 中 Moonlight-Instruct 因 8K 上下文上限多项记 "-"，长上下文与 AIME/HMMT/OJBench 对照不完整
- 图 3（架构图）与式 1/8/9 下方的图形示意在 pdftotext 中为不可读字符堆，机制内容按正文与附录 B 伪代码（Listing 1）转写
- 式 8/9 的 $\Gamma^{i\to C}$ 上标在 PDF 文本层有压扁变形，按附录 B Proposition 1/2 的记号定义（§2.1 Decay Formulation）复核后转写
