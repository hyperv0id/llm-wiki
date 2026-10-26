# Ingest 报告：flashattention-2205-14135（FlashAttention）

源：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré（Stanford / Buffalo），NeurIPS 2022。归档版本 arXiv:2205.14135v2（2022-06-24 编译，34 页，含附录 A–E），`raw/flashattention-2205-14135.pdf`（被 .gitignore 排除，不入 git）。

## 创建
- wiki/source-dao-2022-flashattention.md — WHY：每个 raw 源一个摘要页的仓库约定；覆盖问题/方法/复杂度结论/证据/作者自述局限，单源 confidence 取 medium。
- wiki/flash-attention.md — WHY：技术主体页，承载 tiling + online softmax 重缩放 + 反向重算的分阶段数据流、$\Theta(N^2d^2M^{-1})$ 对 $\Theta(Nd+N^2)$ 的 HBM 访问计数、下界、逐表证据与边界；与改依赖集合的稀疏/近似路线（18 号 Sparse Transformer、19 号 Longformer 一类）的区分落在专门小节并交叉链接。

## 修改
- wiki/native-sparse-attention.md — WHY：「优化目标的分野」一节补 IO-aware 路线起点一句并链接，标明 NSA 的 compute/memory-bound 二分与 FlashAttention 的承接关系（source_count 3→4）。
- wiki/activation-checkpointing.md — WHY：新增 Relation to FlashAttention 一节——反向重算是 checkpointing trade-off 在注意力上的实例，但靠减少 HBM 访问同时拿到加速，补足「重算不一定慢」的对照面（source_count 1→2）。
- wiki/logsparse-self-attention.md — WHY：稀疏方法对照表后补精确注意力与改依赖集合两条路线的正交区分，落实任务要求的路线对照（source_count 2→3）。
- wiki/index.md — WHY：Sources 段（source-dag 与 source-dcrnn 之间）与 Techniques 段（flash-gradient-update 之前）按字母序各插一条，刷新 last_updated。
- wiki/log.md — WHY：追加一条 ingest 记录。

## 新建交叉链接
- [[flash-attention]] ↔ [[logsparse-self-attention]]（精确 vs 稀疏路线对照）
- [[flash-attention]] ↔ [[native-sparse-attention]]（IO-aware 路线承继）
- [[flash-attention]] ↔ [[activation-checkpointing]]（重算 trade-off 的注意力实例）
- 页内单向引用：[[moba-attention]]、[[deepseek-sparse-attention]]、[[grouped-query-attention]]、[[multi-head-latent-attention]]、[[paged-attention]]（省显存/稀疏各轴的相邻页面，未改动对方页面）

## 数值备注
- **GPT-2 加速三处口径不一**：摘要只写「3×」；§4 开头贡献列表写「至多 3× 对 HuggingFace、1.8× 对 Megatron」；§4.1 正文与 Table 2 caption 写「对 Megatron 至多 1.7×」；Table 2 表内为 small 对 HF 3.5×、对 Megatron 2.0×，medium 对 HF 3.0×、对 Megatron 1.8×（表内括号与天数比值 4.7/2.7≈1.7、11.5/6.9≈1.7 也对不上）。四处互不完全一致，页面并列照录，不替论文挑数。
- **摘要「6.4 points of lift」**：为 MIMIC-III +4.3（52.8→57.1）与 ECtHR +8.5（72.2→80.7）的算术平均（Table 5 可复算），§4.2 正文分别给两任务数字，页面两处都写。
- **Table 6 基线单元格只印「7」**：Path-X/Path-256 的全部基线格无数点、无百分号、无口径说明；正文仅称基线「OOM 或随机水平」。「7」是 7.x% 截断、排版残缺还是其他口径无法判定，照录原文符号。
- **任务提示数字不在论文中**：ingest 指令举例「GPT-2 1.5B、序列 4k 的省显存倍数」——论文 GPT-2 实验只有 small/medium（124M/355M 级），长上下文实验为 small；显存证据全部来自注意力算子基准（图 3 右的 20×/2× 与 Table 21，20× 出现在 seq 4096 处 17,024 MB 对 836 MB）。页面按论文实际数字落页，未补写 1.5B 数字。
- **pdftotext 转换缺陷**：`-layout` 模式丢失 `$\lceil\rceil$` 与分式（Algorithm 1 行 1 块大小显示为「4Md」）；经 `-raw` 模式核对为 $B_c=\lceil M/(4d)\rceil$、$B_r=\min(\lceil M/(4d)\rceil, d)$，与 Appendix C 证明的 $\Theta(M/d)$、$\Theta(\min(M/d,d))$ 一致。图 5–8（跨硬件加速曲线）为图像，点值不可读，页面只录文字叙述的范围（2–4×、2.5–4.5×、至多 3×）。
- **18/19 号论文页面不存在**：Sparse Transformer 与 Longformer 在本 vault（及主仓库 wiki/）均无页面，任务要求的「与 18/19 号交叉链接」落到本库现有稀疏注意力页 [[logsparse-self-attention]]、[[native-sparse-attention]]、[[moba-attention]]、[[deepseek-sparse-attention]]，正文中以名字提及 Sparse Transformer/Longformer 而不加悬空 wikilink。
- **LRA 口径**：Table 3 注 3 自述 LRA 准确率依赖调参流程，作者对复现失败的基线取 Tay et al. 或 Xiong et al. 报告的更好值；加速比为五任务 wall-clock 几何平均（E.3）。页面照录该口径。
- **长文档分类波动**：Table 5 中 MIMIC-III 在 seq 1024（50.7）低于 512（52.8）、ECtHR 在 16K（79.2）低于 8K（80.7），论文归因于分布偏移的可能；页面按表照录，未做平滑。
