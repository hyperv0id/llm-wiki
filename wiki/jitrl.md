---
title: "JitRL (Just-In-Time Reinforcement Learning)"
type: entity
tags:
  - llm-agents
  - reinforcement-learning
  - test-time-learning
  - continual-learning
  - memory
  - icml-2026
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# JitRL（Just-In-Time Reinforcement Learning）

JitRL 是新加坡国立大学（Bryan Hooi 组，一作 Yibo Li）提出的免训练 LLM agent 持续学习框架，ICML 2026 接收（论文首页印有 Proceedings of the 43rd ICML, Seoul, PMLR 306, 2026；arXiv:2601.18510v3, 2026-06-08）[^src-jitrl]。核心主张：把 RL 的策略改进搬 到测试时——不做任何梯度更新，而是维护一个动态非参数经验记忆，推理时检索相关轨迹估计动作优势（advantage），用闭式加性规则直接修改 LLM 输出 logits，即 [[test-time-policy-optimization]]。论文称这构成一个非参数策略分布，与把记忆当 ICL 文本的方法分立（见 [[non-parametric-policy-memory]]）[^src-jitrl]。

## 解决的问题

LLM agent 部署后权重冻结：遇到不熟悉或动态变化的环境时无法从错误中学习（论文引 Hendrycks et al. 2025 的 AGI Score，称现有系统最缺"持续学习新信息的能力"）。两条既有路线各有瓶颈[^src-jitrl]：

- **梯度 RL**（PPO/GRPO、WebRL 等）：需要大量交互与训练开销，且产出静态模型、易灾难性遗忘；论文引 He et al. 2025 称常规 RL 在持续适配设定下改进有限。
- **ICL/记忆类**（Reflexion、AWM、MemGPT 等）：受上下文长度限制（长交互序列下失效，论文引 lost-in-the-middle），且 prompt 只能承载显式文本描述，缺乏 RL 那种"由奖励优化出难以言传的技能"的通用性。

JitRL 的回答是：保留 RL 的形式化（价值、优势、KL 约束策略改进），把参数更新替换为对 logits 的推理时调制。

## 机制

框架三组件（论文 Fig 2，算法见附录 A）[^src-jitrl]：

### 1. 记忆构建（§4.1）

记忆库 $\mathcal{M}=\{(s_i,a_i,G_i)\}_{i=1}^N$ 存储转移三元组，$G_i$ 是该转移之后的折扣回报，$G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u$（式 3），隐式刻画环境的经验分布。

- **Reflective step-wise rewards**（见 [[reflective-stepwise-reward]]）：长轨迹 credit assignment 困难，episode 结束后由 LLM Evaluator 对整条轨迹回溯打逐步标量奖励 $r_t$（WebArena 用 −3..+3 分档、含确定性评级；Jericho 按游戏得分尺度校准）。
- **状态抽象**：原始观测（HTML DOM 树、冗长游戏文本）对检索太吵，抽象为保留任务相关语义的紧凑结构化状态，设计原则是"功能等价的状态映射到相似表示"。WebArena 用正则化 URL（把具体 ID 替换为占位符，使同类页面跨实例迁移）加局部动作历史；Jericho 压缩为 `Step t: [State: 名词...] [Action: 动词...]`，另维护层级上下文（全局 [SUMMARY]、里程碑 [PROGRESS]、剪掉无效回路的 [LOCATION]）。

### 2. 测试时价值估计（§4.2）

不训练价值网络，靠检索即时估计。检索用 token 化状态的 Jaccard 相似度取 top-k 邻域 $N(s)$（WebArena 两级匹配：先按正则化 URL 过滤同页型，再对有效状态算 Jaccard；Jericho 用状态索引与历史索引的加权混合 0.75/0.25，再以 Jaccard 过滤）[^src-jitrl]：

- 状态值（式 4）：$\hat V(s)=\frac{1}{|N(s)|}\sum_{i\in N(s)}G_i$；
- 动作值（式 5）：对动作匹配子集 $N(s,a)=\{(s_i,a_i,G_i)\in N(s):a_i=a\}$ 取均值；
- **未见动作**：按概率 λ 走 optimism under uncertainty——$\hat Q(s,a)=\hat V(s)+\alpha/|N(s)|$（式 6），记忆覆盖越稀疏 bonus 越大，经验积累后自然转向利用；以概率 1−λ 置 0 防止过度探索。这是 [[exploration-vs-exploitation|探索-利用权衡]] 在测试时检索层面的实例；
- 优势（式 7）：$\hat A(s,a)=\hat Q(s,a)-\hat V(s)$，即相对局部平均的超出量，作为真优势 $A^\pi(s,a)$ 的代理。

### 3. 策略更新（§4.3）

- **候选集扩充**（式 23）：$C\leftarrow C_{\text{LLM}}\cup\{a_i:(s_i,a_i,G_i)\in N(s)\}$，防止模型忽略历史上有效的动作；仅来自记忆的动作基 logit 置 0。WebArena 动作做语义归一化（临时元素 ID 映射到 accessibility tree 描述）；Jericho 把输出约束在游戏引擎给定的合法动作集内。
- **闭式 logit 更新**（式 10/12）：$z'(s,a)=z(s,a)+\beta\cdot\tilde A(s,a)$，其中优势按 $\tilde A=A/(\max_{a'}|A(s,a')|+\epsilon)$ 归一化（式 25）保证数值稳定。对 $z'$ 取 softmax 即恢复最优策略分布 $\pi^*\propto\pi_\theta e^{\beta\hat A}$。
- **黑盒模型适配**：两种 logit 提取变体——Token-level（让模型输出候选编号 token，取其 log-prob）与 Verbalized（让模型显式输出 0–100 置信度再转 logits）。

## 理论（§4.4 + 附录 B/C/D）

三条递进定理构成"估计一致 → 更新最优"的链条（见 [[kl-regularized-policy-optimization]]）[^src-jitrl]：

1. **Theorem 4.1（更新的最优性）**：加性 logit 更新 $z'=z+\beta\hat A$ 是约束问题 $\max_{\pi'}\mathbb{E}_{a\sim\pi'}[\hat A(s,a)]-\frac{1}{\beta}D_{\mathrm{KL}}(\pi'\|\pi_\theta)$ 的精确闭式解。附录 B 用 Lagrangian 推导：$\log\pi'(a)=\beta\hat A(s,a)+\log\pi_\theta(a)+\text{const}$，归一化后即式 (9)。
2. **Theorem 4.2（估计的 tracking 一致性）**：在非平稳策略序列 $(\pi_t)$ 下（记忆由不同时期的策略生成），固定查询对 $(s,a)$ 有 $\hat V_t\xrightarrow{p}V^{\pi_t}$、$\hat Q_t\xrightarrow{p}Q^{\pi_t}$、$\hat A_t\xrightarrow{p}A^{\pi_t}$。假设（附录 C）包括：状态正则性（V/Q 对状态距离 Lipschitz）、噪声模型（$G_i=Q^{\pi_{t(i)}}+\epsilon_i$，零均值、方差有界）、kNN 机制（$k\to\infty$、$k/N\to 0$、支撑覆盖）、动作频率 $k_a\to\infty$、慢策略漂移 $\Delta_t\to 0$、策略正则性（对策略的 TV 距离 Lipschitz）。证明把误差分解为**状态失配**（邻域平均距离 →0）、**策略漂移**（$\Delta_t\to 0$）与**方差**（$\sigma^2/k_a\to 0$）三项，逐项消失。
3. **Theorem 4.3（策略更新的一致性）**：$\hat\pi_t(\cdot|s)\xrightarrow{p}\pi_t^*(\cdot|s)\propto\pi_\theta\exp(\beta A^{\pi_t})$，对有限候选动作集由连续映射定理直接得到。

## 论文报告的实验

**协议**：多轮次序贯测试——每个任务连续执行 L=5 次（WebArena）/50 episodes（Jericho），报告 Avg（全部尝试均值，反映学习效率）与 Final（末次成绩，反映收敛能力）；Final−Avg 差距越大说明学习越陡[^src-jitrl]。JitRL 与 training-free 基线用 Gemini-2.5-flash；WebRL/SFT 用 Llama-3.1-70B-Instruct 官方 checkpoint；Jericho 的 GRPO 用逐游戏训练的 Qwen3-32B[^src-jitrl]。

### RQ1 主结果

- **WebArena**（Table 1，5 站点微平均）：JitRL 46.98/51.35（Avg/Final），全面高于 Static 35.63/36.30、Memory 41.36/43.00、Reflexion 41.08/42.12、AWM 39.37/40.32、EvoTest 39.24/42.49；结构化、轨迹可复用性高的 Shopping 提升最大（论文称较 Static +73.2%，41.67 对 24.06）；Reflexion 在 Map 出现"reflection noise"（误导性反馈拖累成绩）[^src-jitrl]。
- **WebArena-Lite（held-out）vs 权重更新**（Table 2）：SFT 23.00、WebRL 46.06、JitRL 60.00（Final 成功率）。注意这是**跨骨干对照**（JitRL 用 Gemini-2.5-flash，WebRL 用 Llama-3.1-70B），WebRL 在其余 WebArena 任务上训练[^src-jitrl]。
- **Jericho**（Table 3，Game Score，Avg/Final）：JitRL Library 25.9/30、Zork1 53.0/69、Zork3 3.1/5，三游戏全部第一；最强基线 EvoTest 21.5/26、46.8/54、2.6/4；GRPO 13.6/11、16.2/10、1.1/2。学习曲线（Fig 3）：JitRL 前 10–15 episodes 即有竞争力、差距随 episode 扩大、后期方差收窄；GRPO 全程高方差（稀疏奖励下梯度更新不稳）；Memory/AWM 早期即平台化（论文归因于过度依赖既有记忆模式、抑制探索）[^src-jitrl]。

### RQ2 泛化

- **跨骨干**（Table 4）：GPT-5-mini 与 DeepSeek-V3.2 上 JitRL 大多数情况最优（如 DeepSeek-V3.2 Admin 50.65/54.35），论文称 logit 更新机制 model-agnostic[^src-jitrl]。
- **跨任务冷启动**（Table 5）：只允许从不相交任务检索记忆时 JitRL 仍全面领先（Shopping 36.98 对 Static 23.44），论文解读为迁移的是抽象程序性知识而非具体解。Table 6 显示跨任务记忆占检索上下文平均 47.03%（论文称约一半）[^src-jitrl]。

### RQ3 定性分析

Table 7（WebArena）与 Table 17（Jericho）展示检索记忆如何在 logit 层面压低错误直觉、抬升正确选项：评测页在 Marketing 而非 Catalog（0.70→1.40 对 0.90→0.40）；全局搜索改走 Forums；click(Products) 改 hover(Products)（0.40→0.90）；Library 里"give ID to attendant"（+5 分并解锁 rare books room）；Zork1 Loud Room 反直觉的 `echo` 消音后才能取铂条；Zork3 悬崖必须先 `tie rope to railing` 再下降[^src-jitrl]。

### RQ4 消融

- **Logit 更新 vs Prompt 更新**（Table 8）：同样的检索内容改放 prompt 中，Admin/Reddit 49.46/53.02 对 logit 更新 52.31/57.64——论文归因于长上下文中模型难以稳定关注检索线索，而 logit 调制直接改输出分布[^src-jitrl]。
- **检索邻居数 k**（Fig 4）：8–14 稳健；过小则证据不足方差大，过大引入噪声。
- **探索率 λ**（附录 N）：Jericho 最优 0.65（λ=0 时 20.8/42.1/2.1 明显更差）、WebArena 最优 0.05（λ=0.75 时降至 41.32）——结构化动作空间需要的探索远少于组合爆炸的文本游戏动作空间。
- **UCB bonus α**（附录 O）：α=5 在两个基准上整体最平衡（α∈{3,5,7,10}）。
- **状态表示**（附录 L）：结构化文本优于稠密嵌入——嵌入把拓扑不同但描述相似的状态判为高相似，检索错配记忆引入噪声。

### 成本（Table 9 + 附录 R）

Static $200、Memory $230、Reflexion $220、AWM $250、EvoTest $220、**WebRL ≈$9,900**、JitRL $290（training-free 方法按 OpenRouter API token 计费）。WebRL 成本分解：Llama-3.1-70B 训练共约 154 小时 × 16×H200 双节点 × $64/小时 ≈ $9,856——SFT 10h，8 个 RL 阶段中任务生成 8h、在线 rollout 48h、ORM 奖励标注 2h、actor-critic 优化 86h[^src-jitrl]。

### 同骨干受控对照（附录 I/J/Q）

这是论文里最值得注意的诚实部分，三组设定结论不同[^src-jitrl]：

- **8B、on-the-fly**（附录 I，Table 12）：Llama-3.1-8B-Instruct 同骨干同数据（每任务 5 次尝试的轨迹），JitRL 存记忆检索 vs WebRL 在线更新，WebArena-Lite Final 成功率 32.97 对 27.27——低数据在线场景 JitRL 胜，论文归因于小样本下梯度更新的不稳定性。
- **70B、分离训练/评测**（附录 J.1，Table 13）：同用 Llama-3.1-70B，JitRL 离线收集记忆（无梯度）、评测期不 adaptation：JitRL 40.88（$200）对 SFT 23.00（$640）、WebRL 46.06（$9,900）——JitRL 逼近 WebRL 而成本约 1/50，但**该设定下 WebRL 更高**；论文明确注明此设定（大离线语料、无测试适配）偏向权重更新方法，JitRL 主打的是拿不到离线数据集的持续学习 regime。
- **GRPO 超参扫描**（附录 Q，Table 24）：Qwen3-32B 在 Zork1 上扫 rollout/batch/PPO epochs/lr，最优配置（8/8/1/1e-5）验证均值 40.7、最高 55，可逼近 JitRL（同骨干均值 53.0，主表 Gemini 骨干 53.0），但该配置 50 步消耗 50×8×8=3,200 条训练轨迹，是 JitRL（恰好 50 条）的 64 倍；主文 GRPO 设定（8/1）也要 400 条（8 倍）。

### 规模性（附录 P，Table 23）

Library 50 episodes、按 500 条分桶：记忆从 0–500 涨到 2000–2500 条时检索延迟 15–22ms → 47ms，平均分 18.1 → 30.0——性能随记忆持续提升，检索开销相对 LLM 推理可忽略。

## 超参数（附录 H）

Jericho：step limit 60/episode、50 episodes、温度 0.8、候选数 |C|=3、折扣 γ=0.5、k=10、相似度阈值 0.95、λ=0.65、α=5。WebArena：step limit 10、温度 0.8、|C|=3、γ=0.1、k=10、相似度阈值 0.8、任务相似度阈值 0.27、历史/任务权重 0.7/0.3、λ=0.05、α=5（Table 11 另列 Number of episodes 50，与正文 L=5 的口径未在论文中调和，本页以正文协议为准）。两基准 γ/λ 的量级差异与任务结构相关：WebArena 步数短、动作空间结构化（γ=0.1、λ=0.05），Jericho 长 horizon、动作空间组合爆炸（γ=0.5、λ=0.65）[^src-jitrl]。

## 局限（论文自述，Limitations 章节）

1. **依赖冻结基模型**：JitRL 只能在基模型提出的候选集上重加权，无法发现基模型根本不会生成的动作；且依赖 LLM evaluator 的逐步奖励——credit 归因错误会直接污染优势估计。
2. **不适配难文本化的任务**：状态表示与检索都在文本上操作，空间推理（棋盘局面）、时间序列预测等关键模式难以文本表达的任务，文本检索抓不住状态相似性。
3. Impact Statement 另指出：记忆库存真实交互轨迹，真实部署中可能捕获敏感用户信息（隐私风险）[^src-jitrl]。

## 相关页面

- [[test-time-policy-optimization]] — JitRL 所属的问题类与三条路线对比
- [[non-parametric-policy-memory]] — "记忆即非参数策略分布"的核心概念
- [[kl-regularized-policy-optimization]] — 闭式解背后的目标函数
- [[reflective-stepwise-reward]] — LLM evaluator 逐步奖励技术
- [[in-context-learning]] — 被对比的 ICL 路线
- [[exploration-vs-exploitation]] — optimism bonus 的理论母题
- [[action-value-function]] — Q/V/A 的 RL 定义
- [[grpo-for-forecasting]] — GRPO（梯度路线对照）
- [[ts-memory]] / [[parametric-memory-distillation]] — 另一种记忆利用方式：把检索知识蒸馏进参数、推理时免检索
- [[continual-spatio-temporal-forecasting]] — 时空领域的持续学习问题（同为"部署后继续学习"）
- [[source-jitrl]] — 源摘要

[^src-jitrl]: [[source-jitrl]]
