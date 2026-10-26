# Ingest 报告：minimax-sparse-attention-2606-13392

## 创建
- wiki/source-minimax-2026-sparse-attention.md — WHY：单源摘要页，落问题/机制/证据/局限与原始口径（300–500 字）
- wiki/minimax-sparse-attention.md — WHY：机制主页面——Index/Main 两支路、KL 训练三机制、分阶段数据流与维度变换、kernel 设计、Table 1–6 与 Figure 2–4 数字回打、稀疏注意力谱系区分、论文自述边界

## 修改
- wiki/native-sparse-attention.md — WHY：相关页面加反向链接（谱系族内互指）
- wiki/moba-attention.md — WHY：相关页面加反向链接
- wiki/deepseek-sparse-attention.md — WHY：相关页面加反向链接
- wiki/compressed-sparse-attention.md — WHY：相关页面段落加反向链接
- wiki/longformer-attention.md — WHY：「与相邻稀疏注意力工作的区分」段末加反向链接
- wiki/index.md — WHY：Sources / Techniques 段各加一条，按字母序插入

## 新建交叉链接
- [[minimax-sparse-attention]] ↔ [[native-sparse-attention]]
- [[minimax-sparse-attention]] ↔ [[moba-attention]]
- [[minimax-sparse-attention]] ↔ [[deepseek-sparse-attention]]
- [[minimax-sparse-attention]] ↔ [[compressed-sparse-attention]]
- [[minimax-sparse-attention]] ↔ [[longformer-attention]]

## 命名理由
- 机制页名 `minimax-sparse-attention`：与库内 DSA → `deepseek-sparse-attention` 的「机构前缀 + 方法全名」命名族一致；不用缩写 `msa` 作页名（与 masked self-attention 等常用缩写冲突）；核心机制（per-GQA-group blockwise Top-k + KL 训练索引）作为页面副题与正文主线。任务提示中的「msm」按 arXiv 原文判定为「msa」笔误。
- 脚注键取 raw 文件名 slug `src-minimax-sparse-attention-2606-13392`，页面名按任务指定用 `source-minimax-2026-sparse-attention`（键与页名不逐字符一致，系任务指定页名与 slug 规则的折中，特此说明）。

## 清单核对
- 官方 arXiv:2606.13392v2（2026-06-12）标题、作者（Xunhao Lai 等 17 人）、机构（MiniMax + 北京大学/NVIDIA/浙江大学/华中科技大学/南京大学/杭州电子科技大学）与任务清单（MiniMax Sparse Attention / MiniMax）一致。

## 数值备注
- 正文与表格不一致一处：C.2 Table 5 中 RULER-32K 无强制 61.5 对强制 65.8（差 4.3 分），正文写「Long-context retrieval is also comparable」（§C.2）；页面两处并列，不替论文取舍。
- Figure 2/3 与 Figure 7–14 的曲线/柱状数值图内不可读；Figure 3 的 recall 仅能读出 0.70–0.95 轴带与「分数 recall 高于块 recall」的相对关系，正文相应表述为定性。
- $d_{idx}$（索引单头维）与 KL 系数 $\lambda$ 论文未给数值，页面标注「无数值」。
- C.2/C.3 消融（Table 5/6）未说明所用模型规模；C.1 明确「更少训练迭代 + 评测子集」。Table 6 的 RULER-8K 84.1/83.0 高于主表 Table 2 同名行区间下沿，照录不调和。
- pdftotext -layout 转换完整，未发现表格缺列或图内数值伪影；效率图（Figure 4）的三个倍数（28.4×/14.2×/7.6×）取自图内标注与摘要/§5.4 正文，两者一致。
