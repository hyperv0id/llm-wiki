# Ingest 报告：PagedAttention / vLLM（arXiv:2309.06180）

## 通读范围与版本

- arXiv v1（2023-09-12），PDF 16 页：正文 §1–§10 至 p.14，p.15–16 为参考文献。**无附录**。
- 已读：摘要、§1 引言、§2 背景（§2.1 自注意力 / §2.2 prompt 与自回归两阶段 / §2.3 细粒度批处理）、§3 内存挑战与 §3.1 现有系统、§4 方法全文（§4.1 PagedAttention、§4.2 KV cache manager、§4.3 解码走查、§4.4 并行采样/束搜索/共享前缀/混合解码、§4.5 调度与抢占、§4.6 分布式执行）、§5 实现（§5.1 三个融合 kernel、§5.2 fork/append/free）、§6 评测（§6.1 设置、§6.2 基本采样、§6.3 并行采样与束搜索、§6.4 共享前缀、§6.5 聊天）、§7 消融（§7.1 kernel 微基准、§7.2 块大小、§7.3 换出 vs 重算）、§8 讨论、§9 相关工作、§10 结论。
- 图表全读：Fig 1–19（Fig 2 按列拆 PDF 文本 + 200 dpi 渲染核对标签归属，Fig 12/14/16/17 曲线数值不可读，只取正文口径），Table 1（全文唯一表格）。

## 创建

- `wiki/paged-attention.md` — WHY：库内 `kv-cache-compression` 只覆盖「压掉/淘汰哪些 KV」，没有页讲 KV cache 的放置与碎片；本页承载分页机制（含 Stage 1–5 维度块与内存浪费的算式），并明确它不是压缩的子类。
- `wiki/source-kwon-2023-pagedattention.md` — WHY：每个 `raw/` 源对应一份摘要；覆盖核心论点、机制、证据与局限，供其他页面引用。
- 新增归档 `raw/pagedattention-vllm-2309-06180.pdf`（`.gitignore` 排除 `*.pdf`，下载件先落 `.tmp/`）。

## 修改

- `wiki/kv-cache-compression.md` — WHY：补「放置轴（与压缩正交）」一节，说明 paging 只改放置、不改每 token 字节数，可与压缩淘汰及 MQA/GQA 头共享叠加；相关页面加 [[paged-attention]]；source_count 5→6，last_updated 2026-10-26。
- `wiki/index.md` — WHY：Techniques 段加 `paged-attention`（按字母序置于 `ordinal-temporal-fusion` 之后），Sources 段加 `source-kwon-2023-pagedattention`（置于 `source-kite` 之后）。
- `wiki/log.md` — WHY：追加一条 ingest 记录（问题 / 机制 / 关键数字 / 边界 / 数值备注）。

## 新建交叉链接

- [[paged-attention]] ↔ [[kv-cache-compression]]（放置轴 vs 压缩/淘汰轴，两个方向都有链接）
- [[paged-attention]] ↔ [[source-kwon-2023-pagedattention]]
- [[paged-attention]] → [[native-sparse-attention]]、[[attention-sink]]（同为「不压字节」的相邻路线）

## 命名决定

页面取名 `paged-attention` 而非 `paged-kv-cache`：论文的贡献是块表 + 注意力 kernel 的联合设计（块表本身不解决问题，kernel 必须能按块 gather 才能让块散落成立），且 PagedAttention 是该技术在文献与实现中的通行名；`paged-kv-cache` 更像一种存储格式，会把「放置」误读成一种压缩/编码方案。机制中的内存管理器部分（block manager、引用计数、抢占）在页内以「机制」小节承载，不另起页。

## 数据流块与浪费比例的逐行核对

- Stage 1–5 的维度与论文对应：块表 $[N_{\text{req}}, N_{\text{logical}}]$ 与 #filled 计数（§4.2、Fig 6）；新 token 落在 $j=\lfloor t/B\rfloor$、$b=t\bmod B$，块满后新取块（§4.3 步骤 ②③）；块级 gather 与一个 warp 读一块（§5.1）；跨块归一化的式 4（$A_{ij}$ 分母 $\sum_{t=1}^{\lceil i/B\rceil}$、$o_i=\sum_j V_jA_{ij}^\top$）；引用计数与写时复制（§4.4 Fig 8、Fig 9 的块释放）。
- 浪费比例核对：Fig 2 的柱状标签按 y 坐标重排后为 Orca (Max) 20.4 / 13.3 / 57.3 / 8.9，Orca (Pow2) 26.8 / 17.9 / 13.6 / 41.6，Orca (Oracle) 38.2 / — / 25.2 / 36.6，vLLM 96.3 + 细窄条。由 token states 反推：$100-20.4=79.6$、$100-26.8=73.2$、$100-38.2=61.8$，即「浪费 60%–80%」这一口径，与 §3.1「只有 20.4%–38.2% 用于存真实 token 状态」逐项对上。
- 代入 OPT-13B 参数复算：单 token KV $=2\times5120\times40\times2\ \text{B}=800$ KB；12 GiB / 800 KB $=15{,}728$ 槽，与 Table 1 的 15.7K 一致；$15{,}728/2048=7.68$，即满长预留只装 7 条请求；ShareGPT 均值 $161.31+337.99=499.3$ token，$499.3/2048=24.4\%$，与 Fig 2 的 20.4% 同量级（时间加权 + 数据集混合解释差距，见数值备注）。分页残量上界 $(B-1)/L_r=15/499\approx3.0\%$，对应 vLLM 的 96.3%。

## 数值备注

- **无附录**：任务书提到的附录（block table 算法、kernel、分布式扩展、实验表）在本版本不存在；对应内容分布在 §4.2–§4.3（块表与 Fig 6）、§5.1（kernel）、§4.6（分布式）、§6–§7（实验）。
- **全文只有 Table 1**，性能数字全在 Fig 2、Fig 12–19；没有 Table 2/3，引用时一律记图号。
- **Fig 2 的 vLLM 柱**：96.3% 之外只剩 3.7% 的细窄条，三分量归属不可读，页面表格里如实写「图中只剩细窄条，分量不可读」，不拆分。
- **Fig 2 与长度推算差 4 个百分点**：Fig 2 的 20.4% 是时间平均、且 caption 只写「§6.2 的实验」，未说明用的是 ShareGPT、Alpaca 还是两者混合；按 ShareGPT 均值算的完成时刻效率是 24.4%。两处并列，不替论文取一个。
- **Table 1 的容量单位**：只有把 12 / 21 / 264 GB 读作 GiB，13B（800 KB/token）与 175B（4.5 MiB/token）的槽数才复现出 15.7K 与 60.1K；66B 按 OPT-66B 的 $d=9216$、64 层算得 9,557 槽，表里写 9.7K，属四舍五入差异，页面按表值引用。
- **加速比口径不同**：§9 与摘要称相对 Orca 2–4×，§6.2 给的是同延迟下可承受请求率 1.7×–2.7×（对 Orca Oracle）/2.7×–8×（对 Orca Max），Fig 13 的并发批是 2.2×/4.3×。三者按各自口径分别写，不互相换算。
- **Fig 12/14/16/17 无可读刻度**：归一化延迟只到 0–1，正文也未给绝对延迟，故页面只报告正文明确写出的比值，不补绝对数值。
- **Fig 3 的槽位口径**：图内标签是「7 个（A）/ 3 个（B）prompt KV 状态」与「2038/507 槽永不使用」；10/5 是 2048−2038、512−507 推得的总占用（含已生成与预留槽）。页面按图内标签表述。
- **§7.3 的 20% 原句歧义（正文与图形张力）**：原文 "recomputation overhead is never higher than 20% of swapping's latency" 按字面是「重算 ≤ 换出延迟的 20%」，与同段「块大时换出更高效」矛盾；更可能想表达「至多比换出慢 20%」。Fig 19a 曲线数值不可读，无法从图裁决，页面按字面转写并注明矛盾。
- **公式编号**：§2.1 注意力公式排版编号为 (3)，但 §2.1 与 §4.1 正文均称 "Eq. 4"；§4.1 分块改写式排版编号恰为 (4)。页面「式 4」指 §4.1 的分块注意力式。
- 页内标注 `[INFERENCE]` 的有四处，均为本页推算而非原文陈述：$499.3/2048=24.4\%$ 的效率上界、$(B-1)/L_r$ 的残量上界、Fig 2 三分量随基线策略变化的成因，以及 Stage 4 跨块 online softmax 的归并方式（论文只写「一个 warp 读一块、融合读出与注意力」，未展开归并细节）。
