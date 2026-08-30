---
title: "TriAttention"
type: entity
tags:
  - llm
  - kv-cache
  - attention
  - rope
  - efficient-inference
  - icml-2026
created: 2026-08-30
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# TriAttention

TriAttention 是一种训练无关（training-free）的 KV cache 压缩方法，由 MIT、NVIDIA 与浙江大学团队提出（共同一作 Weian Mao、Xi Lin、Wei Huang；作者含 Song Han、Bohan Zhuang、Yukang Chen），ICML 2026 接收（regular，OpenReview Submission 10803）[^src-triattention][^src-triattention-openreview]。方法建立在两个前提上：[[qk-concentration]]（pre-RoPE 空间中 Q/K 向量集中于固定中心）与 [[attention-distance-preference]]（集中使注意力 logit 近似为 Q-K 距离的三角级数，距离偏好可由中心预测）[^src-triattention]。

## 解决的问题

长推理（extended reasoning）生成数万 token，KV cache 线性增长成为内存瓶颈。主流压缩方法在 post-RoPE 空间用近期 query 的注意力分数估计 key 重要性，但 query 随位置旋转，方向未过时的代表性 query 极少，观察窗口极小；本应重要但近期未被注意的 token（论文举 retrieval heads 中长期休眠的 token 为例）可能被永久驱逐，破坏思维链[^src-triattention]。问题域的完整梳理见 [[kv-cache-compression]]。

## 打分函数

对 cache 中每个 key $k$（取其 pre-RoPE 表示），打分由两部分组成[^src-triattention]：

- 三角级数分数（式 6）：以校准得到的 Q 中心 $\mathbb{E}[q_f]$ 代理未来 query，
  $$S_{\text{trig}}(k,\Delta)=\sum_f \|\mathbb{E}[q_f]\|\cdot\|k_f\|\cdot\cos(\omega_f\Delta+\phi_f),\quad \phi_f=\arg(\mathbb{E}[q_f])-\arg(k_f)$$
  捕获 [[attention-distance-preference|距离偏好]]；
- 范数分数（式 8–9）：$S_{\text{norm}}(k)=\sum_f (1-R_f)\cdot\mathbb{E}[\|q_f\|]\cdot\|k_f\|=\sum_f(\mathbb{E}[\|q_f\|]-\|\mathbb{E}[q_f]\|)\cdot\|k_f\|$，其中 $R_f$ 是频带 $f$ 的 Mean Resultant Length（见 [[qk-concentration]]）。$(1-R_f)$ 权重实现自适应平衡：集中强时 $S_{\text{trig}}$ 主导，集中弱时范数项补位。

总分为 $S(k,\Delta)=S_{\text{trig}}+S_{\text{norm}}$（式 10）。由于 key 未来会被任意位置的 query 检索，论文在几何间隔的未来偏移集 $D=\{1,2,4,\dots,2^{16}\}$ 上取平均得 $\tilde{S}(k)$（式 11）[^src-triattention]。

## 工程细节

- **窗口式剪枝**：每 128 个生成 token 触发一次；仅当 cache 超出预算 $B$ 时对全部 key 打分并保留 top-B（遵循 R-KV 的协议，默认预算 2048 token；DS-Llama 与 MATH 500 用 512）[^src-triattention]。
- **GQA 处理**：每个 KV 头被 $G$ 个 query 头共享，各头打分尺度不同，故先在每个 query 头内做 z-score 归一化、再跨头取最大聚合（式 12–13）。一个 key 只要被任一 query 头认为重要即保留[^src-triattention]。
- **离线校准**：Q 统计量（$\mathbb{E}[q_f]$、$\mathbb{E}[\|q_f\|]$、$R_f$）从校准数据离线计算。论文报告对校准规模（50k–960k token，45.4–45.8%）与质量（Google 首页 HTML 46.2% ≈ ShareGPT 46.7%，AIME24）均不敏感；用 coding 数据校准、reasoning 数据测试的跨域结果与同域校准相当（Table 3C）[^src-triattention]。

## 论文报告的实验

- **推理基准**（Table 1，同一 KV 预算，AIME24/AIME25）：Qwen3-8B 上 TriAttention 42.1/32.9，Full Attention 57.1/40.8，R-KV 25.4/17.5，SnapKV 34.6/20.0；DS-Llama-8B 33.8/19.6；DS-Qwen-7B 42.5/30.0；GPT-OSS-20B 59.2/49.2。四个模型上均为压缩方法中最优[^src-triattention]。
- **预算扫描**（Fig 5，Qwen3-8B，512–4096）：全预算区间优于 R-KV，低中预算优势最大；AIME25 预算 4096 时 43.3%，超过 Full Attention 的 40.8%[^src-triattention]。
- **吞吐**（Table 4，单张 A100 80GB，16K 解码平均）：AIME25 与 Full Attention 同精度（40.8%）时 563.5 对 222.8 token/s（2.5 倍）或 KV 内存 10.7 倍缩减；MATH 500 同精度档（68.4% 对 69.6%）1405.2 token/s（6.3 倍，预算 1024）；AIME24（54.6% 对 57.1%）1.9 倍（预算 4096）[^src-triattention]。
- **对 R-KV 的对照**（Table 5）：同精度下 KV 预算减半（1024 对 2048）且吞吐约 +85%（1405 对 760 token/s）；同预算下 MATH 500 +8.0、AIME24 +15.4 个百分点[^src-triattention]。
- **递归状态查询基准**：论文自建 benchmark，用 DFS 递归模拟（报告当前节点、栈状态、已访问集合，栈完全匹配为指标）测 KV 剪枝对中间状态保持的影响。Qwen3-8B、步数 6–20、每档 80 样本、预算 2048：depth≤16 时与 Full Attention 相当（depth 8/12 略超），depth 18 起落后；R-KV 在 depth 14→16 间从约 61% 骤降至 31%[^src-triattention]。
- **通用任务**（附录 E–F）：LazyEviction 官方框架下 AIME24（DS-Qwen-7B）按 10%/20%/30% 预算得 40.0/43.3/46.7，全档高于 LazyEviction（33.3/40.0/43.3），30% 档追平 FullKV 46.7；LongBench 16 子任务平均 48.1（16 项中 11 项最优，Full Attention 47.2，Ada-KV+SnapKV 45.6；H2O 因需 O(n²) 内存、在可运行的 12 个子任务中 10 项被 TriAttention 胜出）；RULER 检索 66.1（SnapKV 55.6、StreamingLLM 61.1）[^src-triattention]。
- **消融**（Table 3/E/F）：去掉 $S_{\text{trig}}$ 只靠范数，AIME24 从 42.1 降至 18.8；去掉 $S_{\text{norm}}$，论文正文报 AIME24 从 45.8 降至 40.4（−5.4）；去掉 $(1-R_f)$ 加权，AIME24 41.3 对 42.1、AIME25 28.7 对 32.9；偏移上限 128→4096 带来 41.7→48.8，几何间隔（45.8）显著优于线性间隔（28.7）[^src-triattention]。
- **部署演示**（附录 J）：Qwen3-32B（AWQ INT4）+ 单张 RTX 4090 24GB 运行多轮 agent 任务 OpenClaw（首个请求已超 15k token），Full Attention 因 KV cache 无界增长 OOM，TriAttention 全程在显存预算内完成任务[^src-triattention]。

## 评审与接收

ICML 2026 四位评审给分 5/4/4/4，初始批评集中于评测域窄（数学推理为主）与基线不全；rebuttal 补交的 LongBench、RULER、LazyEviction、Ada-KV 对比与校准/偏移消融大多并入 arXiv v1 附录，meta 评论称 rebuttal 后评价一致正面、建议接收（regular）[^src-triattention-openreview]。Ct9M 关于"未利用语义重要性"的质疑由范数项回应（见 [[source-triattention-openreview]] 中仅见于 rebuttal 的语义 token 范数论断）；S1NU 对多轮交互/任务混合场景保持部分保留[^src-triattention-openreview]。

## 局限与不一致

- 论文自述（附录 A）：三角级数计算与剪枝流程尚无专门的高性能 kernel，作者将其列为首要后续工作；评测拟扩展到 coding 与 agentic 任务；head-specific 预算留作未来工作[^src-triattention]。
- **内部数字不一致**：Table 1 与 Table 3 中 TriAttention 在 AIME24（Qwen3-8B，预算 2048）为 42.1%，而附录消融（$S_{\text{norm}}$ 消融正文、Table E、Table F）同一配置报 45.8%，论文未解释两者的差异；本 wiki 对两组数字分别归因、不择优采用[^src-triattention]。
- 评审 gCHe 指出方法依赖两处近似：头内统一中心的假设，以及范数项与未来偏移设计偏启发式、其合理性主要靠消融支撑[^src-triattention-openreview]。
- 三角级数重构是近似而非精确还原：逐头重建 Pearson r 峰值 0.6–0.9、均值 >0.5（见 [[attention-distance-preference]]），意味着相当比例的头拟合一般[^src-triattention]。

## 相关页面

- [[qk-concentration]] — 方法所依赖的核心经验现象
- [[attention-distance-preference]] — 三角级数与距离偏好的数学机制
- [[kv-cache-compression]] — 问题域与 post-RoPE 方法谱系
- [[attention-sink]] — 相关现象：初始 token 注意力汇聚
- [[rope]] — RoPE 本身；TriAttention 的分析完全基于其旋转结构
- [[source-triattention]] — arXiv 版摘要
- [[source-triattention-openreview]] — ICML 2026 评审记录

[^src-triattention]: [[source-triattention]]
[^src-triattention-openreview]: [[source-triattention-openreview]]
