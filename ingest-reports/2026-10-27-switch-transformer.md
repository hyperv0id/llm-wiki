# Ingest 报告：switch-transformer-2101-03961

源：Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（Fedus, Zoph & Shazeer, Google）。arXiv:2101.03961**v3**（2022-06-16），JMLR 23 (2022) 1–40（Submitted 8/21; Revised 3/22; Published 4/22）。PDF 归档 `raw/switch-transformer-2101-03961.pdf`（40 页，pdftotext -layout 全文可读，表格与图注均能回打）。

## 通读范围

§1 引言、§2 Switch Transformer（2.1 简化路由、2.2 高效稀疏路由、2.3 组装、2.4 训练/微调技术）、§3 缩放性质、§4 下游结果（微调/蒸馏/多语言）、§5 数据/模型/expert 并行与万亿参数模型、§6 相关工作、§7 讨论、§8 未来工作、§9 结论；附录 A（attention 内 Switch 层）、B（No-Token-Left-Behind）、C（router 探索策略）、D（低算力档）、E（上下游性能关系）、F（Mesh-TensorFlow 伪代码，含负载损失与 router 全码）。

## 创建

- wiki/switch-transformer.md — WHY：任务指定的 technique 主页。问题→机制→证据→范围结构；前向数据流按 5 个分阶段 `aligned` 块写（路由打分 / top-1 选择 / 容量裁剪与派发 / 专家计算与 all-to-all / 合并），每行注解同时给维度变换与物理含义；乘性辅助损失给 $\sum f_e = 1$、均匀路由退化为 $\alpha$ 的推导；与 [[sparse-gated-moe-layer]]、[[gshard]] 的机制差异表（top-1 vs top-2、损失形式、容量因子数值、精度、挂载位置）。
- wiki/fedus-2022-switch-transformer.md — WHY：source-summary（约 460 字，confidence medium），覆盖核心论点、方法四件套、证据与论文自述局限。

## 修改

- wiki/index.md — WHY：Sources 段按字母序插入 source-fedhint 之后；Techniques 段插入 ssdl 与 std-mae 之间（switch-transformer 字母序落点）。
- wiki/cluster-based-gating.md — WHY：动机段点名 Shazeer 2017 top-k 门控处补一句 Switch 谱系反链（top-1 + 乘性辅助损失），并刷新 last_updated。source_count 不变（未新增 [^src-*] 引用）。

## 新建交叉链接

- [[switch-transformer]] ↔ [[cluster-based-gating]]（top-k 线性门控 ↔ top-1 简化）
- [[switch-transformer]] → [[fedus-2022-switch-transformer]]（[^src-switch-transformer-2101-03961] 全部 30 处回打：正文 29 + 页脚定义 1；source-summary 页自身内联引用 18 处 + 页脚定义 1）

## 数据流自洽检查

- 维度链：$[T,d]\xrightarrow{W_r}[T,E]\xrightarrow{\text{softmax}}[T,E]\xrightarrow{\text{Top}_1}[T]\xrightarrow{\text{onehot}}[T,E]\xrightarrow{\text{einsum gather}}[E,C,d]\xrightarrow{\text{FFN}}[E,C,d]\xrightarrow{\text{combine}}[T,d]$，与附录 F 伪代码（Fig. 15/16）逐形状核对一致；$C=(T/E)\cdot c_f$ 为式 (3)。
- 派发 einsum 写成论文式 (7)（§5.4：$[n,B/n,d]\times[n,B/n,E,C]\to[n,E,C,d]$）；页面按无模型并行口径记 $[\bcancel{T},d]\times[T,E,C]\to[E,C,d]$，与伪代码 `mtf.einsum([inputs, dispatch tensor], reduce dims=[tokens per core])` 同一运算。
- 辅助损失：$f_e$ 不可微、$P_e$ 可微（§2.2 正文），乘 $E$ 归一（论文式 (4) 说明）与 $\sum_e f_eP_e\big|_{\text{uniform}} = 1/N$ 自洽。

## 关键数字与出处（回打）

- Table 1（128 expert、32 核 TPUv3、阈值 -1.50）：T5-Base -1.731 未达标；Switch-Base cf1.0 -1.561 / 62.8h / 1000 ex/s 对 MoE-Base cf1.0 -1.572 / 80.1h / 860；Switch-Base+（768→896、14→16 heads）-1.534 / 67.6h / 780。
- Table 2（选择性精度）：float32 -1.718@1160、bfloat16 -3.780 发散@1390、selective -1.716@1390。
- Table 3（初始化）：0.1x -2.72±0.01 对 1.0x -3.60±0.68（32e、3.5k 步、3 seeds）。
- Table 4（expert dropout）：d0.1+ed0.4 GLUE 85.2 / SQuAD 83.7 / SuperGLUE 73.0 / CNN/DM 19.6。
- Table 5（微调，FLOP-matched）：SuperGLUE 79.5 vs 75.1（Base）、84.7 vs 82.7（Large）；Winogrande 73.3 vs 66.6；ARC Chal. 32.8 vs 35.5（唯一下滑项之一）。
- Table 6–8（蒸馏）：-1.444 teacher → -1.580 student（29%）；1.1B–14.7B 压缩 82–99% 保留 37–28%；SuperGLUE 81.3 → 76.6（30%，97% 压缩）。
- Fig. 4–6：2→256e（223M→14.7B）一致改善；64e 7.5x 步数 / 7x 墙钟；对 T5-Large（3.5x FLOPs）2.5x。
- Fig. 7–8（mC4，101 语言 / 107 任务，1M 步）：101 语言全增益、平均 5x step 加速、91% 语言 ≥4x。
- Table 9（250k/500k 步 Neg. Log Perp.）：T5-XXL -1.147/-1.095；Switch-XXL -1.086/-1.008（250k 领先 >0.061、500k 领先 0.087）；Switch-C（1571B、2048e、890B FLOPs/seq、每层专家、15 层）-1.096/-1.043、同算力 4x。
- §5.6（Switch-XXL 503B token 多任务微调）：SQuAD 89.7（SOTA 91.3）、SuperGLUE 87.5（T5 89.3）、ANLI 65.7 vs 49.4；闭卷 NQ 34.4 vs 32.8、WQ 41.0 vs 37.2、TriviaQA 47.5 vs 42.9。
- Table 10（attention 内专家）：float32 可训（@100k -1.524 vs FF -1.548）、bfloat16 发散。
- Table 11（router 探索）：argmax -1.471 / sample softmax -1.570 / input dropout -1.480 / input jitter -1.468。

## 数值备注

1. **任务简报与源不符（4 项，均未写入页面）**：简报列的「Router z-loss」「微调 expert 扩容 +4%」「容量因子微调放大 2.5/5–10」「附录 hierarchical、编码器容量上限推导」在本 v3/JMLR 版全文（含全部附录）均不存在；这些内容属 ST-MoE: Designing Stable and Transferable Sparse Expert Models（Zoph et al., 2022, arXiv:2202.08906）。本文的稳定性手段是 selective precision + 初始化缩放 + $\alpha{=}10^{-2}$ 辅助损失；微调侧的 expert 相关改动是 expert dropout（Table 4）；容量因子数值只出现在预训练侧（2.0 与 1.25/1.0 消融）。为不编造源外内容，页面按源实写，简报对应条目以此报告为准。
2. **Table 11 图注与正文矛盾**：图注写 "(lower is better)"，但表头标 Quality (↑)，附录 C 正文称 input jitter「empirically perform the best」，而 jitter 的值 -1.468 是表中最高值（按论文全局约定 Neg. Log Perp. 越高越好，脚注 4 与各表 ↑/↓ 标记一致）。两处并列报告，不替论文选边；页面在 Stage 1 注记中并存两说（表头/正文的 jitter 最好 vs 图注的 lower is better）。
3. 其余表格与图内数值均可从 PDF 文本层回打，无转换缺陷；图 4/5/6/7/8 的曲线取值以图内标签与正文引用数字为准，未逐点读图。

## 其他

- 谱系锚点 [[sparse-gated-moe-layer]]（Shazeer 2017）与 [[gshard]] 库内尚无对应页面，页面按悬空 wikilink 处理；前作的损失与精度口径取自本文 §2.1–2.2 的转述，GShard 侧细节以其原文为准。
- 主仓库 /home/jcheng/Documents/llm-wiki 在开工时 git status 干净；worktree 分支 ingest/37-switch-20261027，未 commit（按任务约束）。
