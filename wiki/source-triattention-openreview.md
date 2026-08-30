---
title: "TriAttention — OpenReview 记录（ICML 2026, Submission 10803）"
type: source-summary
tags:
  - llm
  - kv-cache
  - openreview
  - icml-2026
  - peer-review
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# TriAttention 的 OpenReview 记录

本页是 [[source-triattention|TriAttention 论文]] 在 OpenReview（forum id 0tgzJK50Jz，ICML 2026，Submission 10803）的评审记录摘要：Program Chairs 于 2026-04-30 给出 Decision: Accept (regular)，页面显示 2026-05-01 发布[^src-triattention-openreview]。

评审构成：四位评审，评分 5（7qmr, Accept）与 4（S1NU、gCHe、Ct9M, Weak accept）；meta 评论称 rebuttal 后四位评审评价一致正面[^src-triattention-openreview]。

初始弱点集中在：实验限于数学推理与递归任务；基线不含 LazyEviction、Ada-KV；离线校准的规模/质量敏感性未分析；未来偏移集 $D$ 的选择缺消融；Ct9M 质疑只靠距离与范数、未利用语义重要性，检索与对话场景距离模式不明显[^src-triattention-openreview]。

作者 rebuttal（2026-03-31）补交：LongBench 16 子任务平均 48.1（16 项中 11 项最优，Ada-KV+SnapKV 45.6）；RULER 检索 66.1（SnapKV 55.6）；在 LazyEviction 官方框架下 AIME24（DS-Qwen-7B）30% 预算 46.7 追平 FullKV、超 LazyEviction（43.3）；H2O 可运行的 12 个 LongBench 子任务中 10 项胜出；校准规模 50k–960k token 精度稳定（45.4–45.8），Google 首页 HTML（46.2）与 ShareGPT（46.7）相当；偏移集消融（max 128→4096：41.7→48.8；几何间隔 45.8 对线性 28.7）；三域 MRL（Math 0.977 / HumanEval 0.979 / ShareGPT 0.980）；MLA（GLM-4.7-Flash）集中度更高。这些结果大多已并入 arXiv v1 附录 E–I[^src-triattention-openreview]。

仅见于 rebuttal 的论断：语义重要 token（实体名、标点、话语标记）的 key 范数系统性偏大，故范数项可充当位置无关的语义显著性代理[^src-triattention-openreview]。

S1NU 在 rebuttal 后仍标记部分未解决，理由是多轮交互与任务混合等真实场景的适应性未直接验证；chairs 的参考文献自动核查标记了两条文献（Hong et al. 2024、Su et al. RoFormer 2024）作者无法核验[^src-triattention-openreview]。

[^src-triattention-openreview]: [[source-triattention-openreview]]
