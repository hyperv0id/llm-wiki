---
title: "Conditional State Update"
type: concept
tags:
  - llm
  - long-context
  - theory
  - state-machine
  - reasoning-trace
created: 2026-09-13
last_updated: 2026-09-13
source_count: 1
confidence: medium
status: active
---

# Conditional State Update（条件状态更新）

本页把 TRACE AS STATE 论文 §3.1 的形式化定义、Appendix A 的最坏情况构造与两个 cut 上的不可区分性论证完整展开，并澄清该定理能说什么、不能说什么。主入口机制讲解见 [[trace-as-state]]，实验数字见 [[trace-as-state-evidence]]，复现细节见 [[trace-as-state-reproduction]]，与相关工作的比较见 [[trace-as-state-positioning]]。

## 一、因果状态更新系统

论文把"逐单元读入、只用当前记忆与新单元更新持久工作记忆"的处理器抽象为**因果状态更新处理器**（causal state update processor）。设 $C=(c_1,\dots,c_n)$ 是 $n$ 个信息单元的有序序列，$c_i$ 是第 $i$ 个单元。处理器在有限状态空间 $S$ 中携带一个**任务状态**（task state）$s_i$，每读入一个 $c_i$ 就套用一条**固定**更新规则 $U$：

$$s_i = U(s_{i-1},\,c_i),\qquad i=1,\dots,n \tag{1}$$

逐项解释：$s_{i-1}$ 是读 $c_i$ 之前的任务状态；$c_i$ 是刚到达的信息单元；$U$ 是不随 $i$ 改变的映射规则；$s_i$ 是处理完 $c_1,\dots,c_i$ 之后到达的任务状态。"因果"的含义是：更新只依赖当前记忆与刚收到的单元，不会回头重读更早的输入。当初始状态 $s_0$ 固定时，整条序列驱动处理器走一条**唯一实现的状态路径**（realized state path）——$s_0$ 定了，后面每一步都被式 (1) 唯一决定（§3.1, Eq. 1; Figure 1A; Table 1 记号表）[^src-trace-as-state]。

记 $m=|S|$ 为状态空间大小，$b=\log_2 m$ 为状态空间的**对数大小**；把一个状态当作整数存储需要 $\lceil b\rceil$ 比特（§3.1, Table 1）[^src-trace-as-state]。

## 二、条件状态更新任务：条件在头还是在尾

把固定的 $s_0$ 换成一个**任务条件**（task condition）$z\in S$，就得到**条件状态更新任务**（conditional state update task）：处理器令 $s_0=z$，对 $C$ 套用式 (1)，返回 $s_n$。同一个任务可以按两种顺序呈现：

- **condition-first** $[z,C]$：先读 $z$，再读 $c_1,\dots,c_n$；
- **condition-last** $[C,z]$：先读完整个序列，最后才知道 $z$。

直觉是：$[z,C]$ 下处理器一开始就知道自己在哪条状态路径上，只需跟踪**当前**状态；$[C,z]$ 下处理器读完 $C$ 才知道初始状态，而此时已经无法回头重放序列，只能保留足以区分各种初始状态取法的工作记忆——注意这不必是某种结果的显式整表，任何可区分、可压缩的编码都行。Figure 1B 的说法是：condition-first 只需跟踪实现的状态，condition-last 在最坏情况下可能需要指数级更多的工作记忆（§3.1; Figure 1B; Eq. 2）[^src-trace-as-state]。

### 具体例子：$m=4$ 的状态空间

取 $S=\{s_0,s_1,s_2,s_3\}$，$m=4$，$\lceil b\rceil=2$ 比特。"搜索指针当前停在哪一个槽位"是**本页教学例** [INFERENCE]——论文原文只给出 $m=4$ 的比特数检验（Appendix A "As a concrete check"），没有给这个具体故事。构造 $n=1$ 的输入字母表：每个输入单元编码一个函数 $f:S\to S$，含义是"读我之后，指针按 $f$ 移动"。例如单元 $c_f$ 对应 $f(s_0)=s_1,\ f(s_1)=s_0,\ f(s_2)=s_2,\ f(s_3)=s_3$（交换前两个槽位、保持后两个）。$S$ 到自身的函数共 $m^m=4^4=256$ 个，所以字母表有 256 个单元。

- condition-first $[z,c_f]$：先读条件（比如 $z=s_2$），只记下当前指针位置（2 比特）；再读 $c_f$，查表得 $f(s_2)=s_2$。全程 2 比特。
- condition-last $[c_f,z]$：先读 $c_f$，但还不知道 $z$。为了之后对**任何** $z$ 都能答对，必须区分"读到的是 256 个函数中的哪一个"——8 比特。读完 $z$ 后查表输出 $f(z)$。

这就是论文的数值检验：$m=4$ 时 late-order cut 需要 8 比特，condition-first cut 需要 2 比特（Appendix A "As a concrete check"）[^src-trace-as-state]。

## 三、两个 cut 上的不可区分性反证

"cut"指处理过程中一个确切位置：已读前缀、未读后缀。比较工作记忆，就是问：在这个 cut 上，要让处理器在**所有**输入上精确（exact，即输出总与任务定义一致），它的配置（configuration）至少要有多少种可区分取值。两个方向的论证都是反证：若两种输入导向同一配置，则它们之后读到相同后缀必然产生相同输出，但精确性要求输出不同，矛盾（Appendix A）[^src-trace-as-state]。

先定义贯穿全页的量。**残差映射**（residual map）：对输入序列 $C\in\mathcal{C}^n$ 与条件-更新任务 $t:\mathcal{C}^n\times S\to S$，

$$\varphi_C = \bigl(t(C,z)\bigr)_{z\in S}\in S^S,\qquad K=\bigl|\{\varphi_C \mid C\in\mathcal{C}^n\}\bigr|$$

即 $\varphi_C$ 把每个可能的初始状态 $z$ 映到"以 $z$ 出发读完 $C$ 后到达的状态" $t(C,z)$；$K$ 是不同残差映射的个数（Appendix A；编号为本页所加）[^src-trace-as-state]。

**Late-order cut**（读完 $C$、尚未读 $z$）。一般下界：若两个输入序列 $C\neq C'$ 导出不同残差映射 $\varphi_C\neq\varphi_{C'}$，则它们不能在该 cut 共享配置——否则存在 $z$ 使 $t(C,z)\neq t(C',z)$，从共享配置读相同后缀 $z$ 输出相同，违反精确性。故该 cut 至少容纳 $K$ 种配置，精确的 cut 状态需求是 $\lceil\log_2 K\rceil$ 比特；$K$ 取到多大取决于任务，最坏情况见下节构造（Appendix A "Late-order cut"）[^src-trace-as-state]。

**Condition-first cut**（读完 $z$、尚未读 $c_f$）。取恒等映射 $\iota=\mathrm{id}_S$，对应输入 $c_\iota$。若两个不同取值 $z\neq z'$ 在该 cut 产生相同配置，则读相同后缀 $c_\iota$ 后输出必然相同；但精确性要求分别是 $\iota(z)=z$ 与 $\iota(z')=z'$，矛盾。故至少 $m$ 种配置，需求 $\lceil\log_2 m\rceil=\lceil b\rceil$ 比特（Appendix A "Condition-first comparison"）[^src-trace-as-state]。

## 四、一般下界、最坏情况与可达性

把两个 cut 合起来，论文的顺序原理（ordering principle）是：对**任何**条件状态更新任务，condition-last 在"读完序列、尚未读条件"的 cut 上，精确输入依赖状态至少需要 $\lceil\log_2 K\rceil$ 比特（$K$ 为该任务的残差映射个数）；condition-first 只需跟踪单一状态，$\lceil b\rceil$ 比特。注意这是一条**下界**陈述：它约束任何精确处理器在该 cut 的配置数，但一般任务的值没有配套上界（§3.1, Eq. 2; Appendix A）[^src-trace-as-state]。

最坏取值由 Appendix A 的构造给出。恒有 $K\le m^m$（每个 $\varphi_C$ 都是 $S^S$ 中的元素），构造达到 $K=m^m$，此时下界为：

$$\lceil\log_2 K\rceil \;\le\; \lceil\log_2 m^m\rceil = \lceil m\log_2 m\rceil = \lceil b\,2^b\rceil$$

逐项解释：$m^m$ 个自映射各对应一个残差映射，枚举它们需要 $\log_2(m^m)=m\log_2 m$ 比特；代回 $m=2^b$ 得 $b\cdot 2^b$——状态空间每翻一倍（$b$ 加 1），**这个构造**在 late-order cut 的工作记忆从 $\lceil b\rceil$ 比特跳到约 $2^b\cdot b$ 比特，即论文所说 condition-last 最坏情况下"指数级更大"的含义（§3.1, Eq. 2；Appendix A）[^src-trace-as-state]。

**最坏情况构造**（$n=1$，全部自映射字母表）。固定任意非空有限 $S$，令输入字母表为

$$\mathcal{C}=\{c_f \mid f\in S^S\}$$

即每个自映射一个单元，$|\mathcal{C}|=m^m$。定义一条固定更新规则 $U(s,c_f)=f(s)$（等价写法 $U_{c_f}=f$）。对每个 $f$，序列 $C_f=(c_f)$ 属于 $\mathcal{C}^n$，其残差映射满足 $\varphi_{C_f}(z)=t(C_f,z)=U_{c_f}(z)=f(z)$。于是 $\{\varphi_C\}=S^S$，$K=m^m$。任何在该任务族上对所有输入精确、且先读 $C$ 后读 $z$ 的确定性单遍处理器，都必须在读 $z$ 之前保留至少 $\lceil\log_2 K\rceil=\lceil m\log_2 m\rceil$ 比特持久输入依赖状态（Appendix A "Proposition" 与 "Construction"）[^src-trace-as-state]。

**该构造的两个 cut 均紧**。Appendix A 不仅证下界，还给出达到下界的实现：late-order cut 上，把 $f$ 的身份存在 $m^m$ 个状态之一，$z$ 到达时套用固定查表规则即可，精确需求恰为 $\lceil\log_2(m^m)\rceil$ 比特；condition-first cut 上，单个 $S$ 值寄存器（先以 $z$ 初始化、$c_f$ 到达时替换为 $f(z)$）恰有 $m$ 种输入依赖配置，也是最优的。也就是说，在这个构造的任务族上，两个 cut 的下界都是紧的（Appendix A "Late-order cut" / "Condition-first comparison" 中 "attains/attained" 句）[^src-trace-as-state]。

**$m=1$ 边界**。状态空间只有一个元素时，每个 cut 都只有一种配置，需求为零比特——$\lceil\log_2 1\rceil=0$，公式自洽（Appendix A "When $m=1$" 句）[^src-trace-as-state]。

## 五、内存口径：什么算、什么不算

比较工作记忆时，Appendix A 的约定是（Appendix A "Construction"/"Scope"；§3.1 "knows exactly the state update rule $U$"）[^src-trace-as-state]：

- **更新规则 $U$ 不计入**。论证假定处理器"确切知道状态更新规则"，即 $U$ 作为固定知识免费持有；被计数的是**输入依赖**的持久状态。
- **所有辅助可写存储都计入**。计数对象不只是那个 $S$ 值寄存器，而是任何持久、输入依赖的推理缓冲——处理器不能重读 $C$，所以一切"事后补救"手段都必须体现在 cut 时刻的配置里。
- 计的是 cut 处的配置数取对数，即"必须能区分多少种情况"，而不是任何具体实现的字节数。

## 六、定理的适用范围与三个常见误读

Appendix A 的构造是**确定性、精确、单遍、有限状态**任务族上的对抗性最坏情况存在构造：字母表和固定转移表被刻意设计为覆盖 $S$ 的全部自映射（Appendix A "Scope"）[^src-trace-as-state]。由此可以准确复述的边界：

1. **它不推出真实 LLM 的显存差或收益预测。** [INFERENCE] 定理给出的是下界：一般任务在特定 cut 上至少需要 $\lceil\log_2 K\rceil$ 比特，但下界不等于任何实现的精确内存量，因此不能把"$\lceil\log_2 K\rceil-\lceil b\rceil$"读成真实系统可省的比特数。唯一两侧都紧的陈述是 Appendix A 的最坏构造在该 cut 上需要且只需 $\lceil\log_2(m^m)\rceil$ 比特。论文自己只说 §3.1 的结果"motivate"后续实验，没有预言实验收益幅度（§3.2 第一段）[^src-trace-as-state]。
2. **指数差距不是顺序的必然产物，而是更新族丰富度的产物。** [INFERENCE] 举一个受限更新族反例：若字母表只含 $S$ 上的循环移位（$m$ 个置换），则残差映射只有 $m$ 个，$K=m$，condition-last 的下界退化为 $\lceil\log_2 m\rceil=\lceil b\rceil$ 比特——与 condition-first 同阶。Appendix A 明确说受限更新族可实现更少残差映射、得到更小下界（Appendix A "Restricted update families" 句；反例本身为 [INFERENCE]）[^src-trace-as-state]。
3. **随机、近似、多遍计算不在定理覆盖内。** [INFERENCE] 定理只证确定性、精确、单遍情形。随机或近似的处理器能否合并配置需要另行证明：即使随机但零错误，Appendix A 的反证（相同配置 + 相同后缀必须给出相同输出）也不直接适用，"能省"或"不能省"都不能由此定理断言。多遍处理（如 TRACE AS STATE 本身就是第二遍重读 $x$）同样在范围之外，这正是论文用第二遍绕开单遍约束的设计（Appendix A "Scope" 的 deterministic/exact/one-pass 表述；§3.3 的 fresh pass）[^src-trace-as-state]。

## 七、三种"状态"的区分

论文里出现三个层次不同的"状态"，混读会误解整条论证（§3.1；§3.2；Appendix A KV 段）[^src-trace-as-state]：

```
对象 A task state s_i：
    S 中的一个值，式(1)的操作对象，存为整数需 ceil(b) 比特（§3.1）
对象 B complete inference working state：
    当前位置 + 所有层 KV 条目 + 其他持久输入依赖推理缓冲（Appendix A）
对象 C textual trace T：
    推理文本记录，proxy，可能不完整、有损、甚至错误（§3.2; Figure 1C）
```

[INFERENCE] 上图刻意把三个对象**并列**而非用"⊂"或"压缩"箭头连接：论文建立了 $B$ 的因果更新结构与有限性（Appendix A），分别定义了 $A$ 作为任务层抽象（§3.1）与 $C$ 作为文本代理（§3.2），但**没有**给出 A 与 B 之间、或 B 与 C 之间的构造性投影/压缩接口，本页也不代为构造[^src-trace-as-state]。三者的边界：

- **任务状态 $s_i$**：式 (1) 中 $S$ 里的单个值，是形式化论证的操作对象，存为整数需 $\lceil b\rceil$ 比特（§3.1, Eq. 1; Table 1）[^src-trace-as-state]。
- **完整推理工作状态**：Appendix A 把"working state"定义为当前**位置**、**所有层**的 KV 条目，以及任何其他持久输入依赖推理缓冲的总和（Appendix A KV 段）[^src-trace-as-state]。
- **文本 trace 代理 $T$**：推理轨迹的文本记录，**可能不完整、有损、甚至错误**；论文明确不把 $T$ 等同于形式条件 $z$ 或任何特权内部状态，二者的联系只是功能性的——若 trace 含有用状态信息，把它放到上下文前面可以让这些信息在下一遍处理时可用（§3.2 "The proxy may be incomplete, lossy, or incorrect" 段；Figure 1C）[^src-trace-as-state]。

## 八、KV 缓存为什么符合这个抽象

transformer 推理时 KV 缓存随上下文增长，看起来"记忆无限膨胀"，为什么还算有限状态？Appendix A 的论证是：定义工作状态为当前位置 + 全部层 KV + 其他持久输入依赖缓冲；因果推理逐层计算新 token 表示并追加对应 KV 条目，因此下一配置是"前一配置 + 新 token"的固定函数；**最大上下文长度有限、精度有限**，配置空间就是有限的。逐 token 的（或 latent/shared/linear 等变体）缓存因此都不违反因果状态更新抽象（Appendix A "Causal Transformers are Causal State Update Processors" 节）[^src-trace-as-state]。

[INFERENCE] 注意两个易混点。其一，"配置空间有限"是指配置数量有限，**不**意味着单个定长向量：配置包含随已读 token 数增长的变长 KV 序列，有限性来自上下文长度上限与有限精度。其二，"符合抽象"指结构上满足式 (1) 的形式（下一配置 = 固定函数(当前配置, 新输入)）；任务状态 $s_i$ 与完整配置分属**两个层面的抽象**，论文未建立二者之间的构造性对应。论文原文说条件状态更新任务"不是真实长上下文任务的字面模型"（not literal models），它是用来导出可检验预测（条件放前面更好）的思想实验（§3.2 第二段）[^src-trace-as-state]。

## 九、trace 代理的形式化（§3.2）

§3.2 把上述抽象接到实验管线上。设 $M$ 是因果状态更新模型（即被测 LLM），$x$ 是长上下文。对同一问题独立跑 $M$ 共 $n_{tr}$ 次，第 $j$ 次产出一条推理轨迹 $r_j$ 与一个独立的可见答案 $a_j$：

$$(r_j,\,a_j)\sim M(\cdot\mid x),\qquad j=1,\dots,n_{tr} \tag{3}$$

每个任务配一个在各放置条件间**保持不变**的序列化器 $\pi$，构造序列化 trace：

$$T=\pi(r_1,\dots,r_{n_{tr}}) \tag{4}$$

$\pi$ 按源顺序保留纳入的推理文本，加固定标签与分隔符（Appendix C 的 trace-block preamble $P_D$ 属于 $T$ 而非原始任务 prompt $x$；可见答案在 $T$ 之外）。$T$ 就是待检验的文本状态代理，TRACE AS STATE 用新一遍因果 pass 处理 $[T,x]$，对照组 TRACE APPEND 处理 $[x,T]$（§3.2, Eq. 3–4；§3.3；Appendix C.1）[^src-trace-as-state]。与条件 $z$ 的对应关系是：$z$ 是形式化里"已知的任务状态"，而长上下文场景中有用的任务状态往往在读完上下文之后才被发现，trace 恰好是这些状态的文本记录——活动目标、已解析引用、搜索前沿等（§3.2 "The formal condition $z$..." 段）[^src-trace-as-state]。实验端的顺序与截断口径见 [[trace-as-state-reproduction]]，各放置条件的实测结果见 [[trace-as-state-evidence]]。

## 相关页面

- [[trace-as-state]] — 主入口：方法机制与实验总览
- [[trace-as-state-evidence]] — Table 2/3/5 与 Figure 2–6 的完整数字
- [[trace-as-state-reproduction]] — Appendix B/C/F、序列化与 token 统计
- [[trace-as-state-positioning]] — 与 [[source-re-reading-improves-reasoning]]、[[source-state-over-tokens]]、[[source-recontext]]、[[source-core-context-repetition]] 等来源页的关系
- [[source-trace-as-state]] — 主来源摘要
- [[kv-cache-compression]] — 同属"推理期记忆"话题的压缩路线对照

[^src-trace-as-state]: [[source-trace-as-state]]
