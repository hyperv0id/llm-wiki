---
title: "多模态外生信息引导的长期时空预测：研究路线分析"
type: analysis
tags:
  - spatio-temporal
  - multimodal
  - exogenous
  - long-horizon-forecasting
  - causal
  - interpretability
  - research-agenda
  - landscape-analysis
created: 2026-06-08
last_updated: 2026-07-14
source_count: 26
confidence: medium
status: active
---

# 多模态外生信息引导的长期时空预测：研究路线分析

> [!info] 本页定位
> 这是一份**前瞻性研究议程**，不是对既有结论的总结。描述性部分（"现状缺什么"）由知识库多源支撑，置信度高；处方性部分（"该怎么做"）是研究提案，本质是判断与赌注。所有"截至 2026-06 尚无人做 X"的论断都是快照。
>
> **证据基础有两类**：(1) 知识库已 ingest 的源文件 → 用 `[^src-*]` 引用；(2) 用户提供的《2026 AI/ML 顶会论文汇编》（ICLR/ICML/CVPR/AAAI，2048 篇，2026-06-05 快照）中尚未 ingest 的前沿工作 → 列于 [[#外部线索（2026 顶会汇编，尚未 ingest）]]，以 arXiv 号标注、不作脚注。

研究方向定义：利用来自系统外部的多模态信息（气候/天气状态、环境事件、文本报告、图表数据、社会活动等）辅助长期时空系统的演化预测与推理，关注外生多模态信息与时空动态之间的**跨模态关联**与**长期影响机制**，提升长期预测的准确性、泛化能力与可解释性。

---

## 一、中心命题：时空幻象是"消歧"问题，外生信息是未被点名的消歧器

[[spatiotemporal-mirage|时空幻象]]把"相似输入→不同未来 / 不同输入→相似未来"的预测困境归因为**输入窗口太短**，并以 [[std-mae|STD-MAE]] 的 864 步解耦掩码预训练作为解法[^src-2312-00516-std-mae]。这个诊断被框窄了。

> [!note] 重构
> 时空幻象本质是**消歧问题**：当内生历史不足以区分两个未来时，能区分它们的信号——一道锋面、一份事故报告、一张赛事日程——恰恰存在于**外生模态**里。把立论从"窗口不够长"升级为"内生信息不充分"，立刻给出可执行路径：把 STD-MAE 的时间/空间双轴掩码重建扩成**第三轴——跨模态掩码（用外生重建内生、反之亦然）**[^src-2312-00516-std-mae]。

这一重构是整个方向的 motivation 抓手：长期预测的精度天花板，在趋势与周期被捕获之后，主要由**外生信息**这一未开发的杠杆决定。

## 二、中心缺口：四块拼图无人拼合

至今没有任何工作同时做到〔多模态外生融合〕＋〔显式长期影响机制〕＋〔因果去混杂〕＋〔可解释归因〕**并在长 horizon 上验证**。各拼图分别存在：

| 工作 | 已具备 | 缺口（即切入点） |
|---|---|---|
| [[e2-cstp\|E²-CSTP]] (NeurIPS25) | 因果双分支去混杂＋多模态(文/图/ST)[^src-e2-cstp] | 无显式长期衰减；多模态数据集增益薄(Terra +1.6%/BjTT +1.7%)；非长程[^src-e2-cstp] |
| [[igstgnn\|IGSTGNN]]/TIID (KDD26) | 长期影响衰减核 ω_τ=exp(−τ²/2σ²)，长 horizon 增益最大[^src-incident-guided-st-forecasting] | 单模态、仅最新时刻事件、手工空间先验[^src-incident-guided-st-forecasting] |
| [[event-driven-reasoning\|VoT]] (ICLR26) | 首个用 LLM **推理**外生文本、输出推理链[^src-event-driven-ts-forecasting] | 无图结构因果量化、依赖 LLM、非长程数值预测[^src-event-driven-ts-forecasting] |
| [[factost\|FactoST]] | O(N) 解耦、避免负迁移、STMF 元数据槽[^src-factost] | 显式无外生模态；future work 明写"event/text/weather 协变量编码器"[^src-factost] |
| [[visifold\|VisiFold]] | 高效长程(24-48 步)[^src-visifold] | 明写"无法响应突发事件；多模态元数据融入 TFG 是 future work"[^src-visifold] |
| [[swift\|Swift]] | 75 天稳定天气 rollout＋CRPS 集合[^src-swift] | 纯自回归、**无外生条件**、无 classifier-free guidance[^src-swift] |
| [[ctenet\|CTENet]] (NeurIPS25) | ADR 架构嵌入＋欧拉连续空间＋气象外生融合，中美数据集 RMSE −45.8%/−21.0%[^src-ctenet] | 缺因果去混杂、无不确定性量化、计算 O(HW) 高于图方法[^src-ctenet] |
| [[most\|MoST]]/[[aurora\|Aurora]] | 原生多模态/生成式概率预测[^src-most][^src-aurora] | 通用时序非时空图、无长期机制、无因果[^src-most][^src-aurora] |

[[spatio-temporal-foundation-model-landscape|时空基础模型全景]]页自列的三大未解问题正中靶心：**外生事件未建模**（[[conformer|ConFormer]]/IGSTGNN 证明事故剧烈影响交通，但 VoT/TimeCAP 的外生推理模块尚未与 STFM 架构集成）、**概率预测空白**、**天气-交通 FM 谱系能否统一**[^src-conformer][^src-incident-guided-st-forecasting]。换言之，`FactoST` 与 `VisiFold` 自己把这个方向写成了 future work——**这是被官方盖章的空位**[^src-factost][^src-visifold]。

## 三、问题的真实结构（五个硬子问题）

| 子问题 | 为什么难 | 知识库中的相关探索 |
|---|---|---|
| **外生信息异质性** | 模态/采样率/时空粒度/可靠性/**未来可得性**各不同 | [[heterogeneous-covariates\|UniCA 同质化]][^src-unica]、[[most\|MoST SNR 门控]][^src-most] |
| **长期影响机制（科学核心）** | 外生影响是滞后、衰减、累积、状态依赖、分频的，非定点同步 | IGSTGNN-TIID 高斯衰减核[^src-incident-guided-st-forecasting] |
| **跨模态关联 vs 伪相关** | 外生信号含大量虚假相关，长程下模型会"走捷径" | [[e2-cstp\|E²-CSTP 后门调整]][^src-e2-cstp]、[[stop\|STOP DRO]][^src-stop] |
| **三目标张力** | 外生信息**并非总有益**：[[cvpe\|CVPE]] 弱相关数据集 −5.2%[^src-cvpe-2025]、[[middir\|MiDDiR]] 小数据过引导过拟合[^src-middir]、[[tats\|TaTS]] 需 CTR 共振[^src-language-in-the-flow-of-time] | 何时该引入外生模态本身是开放问题 |
| **评测真空** | 无耦合〔外生多模态＋长 horizon 真值＋因果标签〕的基准 | [[accident-aware-traffic-forecasting\|多数 ST 数据集缺事件标注]][^src-conformer]、[[spatio-temporal-reasoning\|配对文本稀缺]][^src-streasoner] |

## 四、统一架构论点

把拼图焊起来，主张一个**"分解骨干 + 晚注入外生适配器"**架构：

```
            [内生骨干: 线性复杂度, 只学跨域不变的时间动态]   ← FactoST/Mamba-3/VisiFold
                         |  (避免负迁移, 为长程×大规模留算力)
        ┌────────────────┴─────────────────┐
        v                                   v
 [外生适配器: 小假设空间, 晚注入]      [因果纯净主干分支]      ← E²-CSTP 双分支
        |  条件-去噪分离/条件重参数化         |  反捷径对比训练 (S-GRPO)
        |  (PriSTI / ConFormer-GLN / adaLN)  |
        v                                   v
 [长期影响 = 分模态·分频·学习型滞后核]  ←─ TIID 高斯核的推广
   慢变外生(气候)→低频/趋势分支
   突发外生(事件)→高频/残差分支         ← HyperD / VoT-AFF / DST-Mamba
        |
        v
 [可靠性闸门: SNR/反馈/有界误差]        ← MoST / FENCE / (MUSE)
        |
        v
 [可解释性近乎免费]: 滞后核(何时起效) + 频带归因(哪模态管哪尺度) + 因果分支(真机制vs伪相关)
```

七条设计原则：

1. **骨干保持内生＋线性复杂度**（[[factost|FactoST]] O(N)）——避免空间负迁移、为长程×大规模留算力[^src-factost]。
2. **外生多模态走独立晚注入适配器**（FactoST 的 STMF 槽即为此设计）——困在小假设空间收紧泛化界[^src-factost]。
3. **注入用"条件-去噪分离"或"条件重参数化归一化"**（[[pristi|PriSTI]] 干净先验图 / [[conformer|ConFormer]] GLN）——别让跨模态对齐被噪声或自回归冲掉[^src-pristi][^src-conformer]。
4. **长期影响 = 可学习的分模态滞后核**：把 TIID 的高斯衰减推广为学习型、分频路由的滞后响应——慢变外生→趋势分支、突发外生→残差分支（[[hyperd|HyperD]]/[[dst-mamba|DST-Mamba]]）[^src-incident-guided-st-forecasting][^src-hyperd-hybrid-periodicity-decoupling][^src-dst-mamba]。
5. **因果双分支兜底**（[[e2-cstp|E²-CSTP]]）＋反捷径对比训练（[[spatio-temporal-reasoning|S-GRPO]] 的 with/without-外生 对比奖励）——确保模型真用了外生信息而非伪相关[^src-e2-cstp][^src-streasoner]。
6. **可靠性闸门**（[[most|MoST]] SNR / [[fence|FENCE]] 反馈强度）——外生信息有噪声、可靠性随 lead-time 衰减[^src-most][^src-fence]。
7. **可解释性近乎免费**：滞后核给"何时"、频带归因给"哪模态哪尺度"、因果分支给"真机制vs伪相关"，再叠 [[event-driven-reasoning|VoT 推理链]][^src-event-driven-ts-forecasting]。

## 五、研究方向（分层；标注 入口路线 A=迭代现有 / B=跨领域；命中目标）

### ⚡ 快赢（低风险、可快速证伪）

- **QW1 · 外生滞后核插件**〔A｜准确性+可解释〕：把 TIID 高斯衰减推广为**学习型、分模态、分频**的滞后响应核，做成即插件挂到 [[visifold|VisiFold]] TF-token / [[factost|FactoST]] STMF / [[dst-mamba|DST-Mamba]] 趋势-季节分支[^src-incident-guided-st-forecasting][^src-visifold][^src-factost][^src-dst-mamba]。**新意 vs TIID**：多模态、多粒度、学习型(非手工)、历史多事件累积。风险低——TIID 已证明衰减机制在长 horizon 增益最大[^src-incident-guided-st-forecasting]。
- **QW2 · 模态价值闸门（fuse 前先筛）**〔A+B｜泛化〕：[[tats|TaTS]] 的 CTR/TT-Wasserstein 先验筛选 + [[most|MoST]] SNR + [[fence|FENCE]] 反馈强度，合成"这个外生模态在这个 lead-time 该不该融、融多强"的控制器，直接回应"外生信息并非总有益"[^src-language-in-the-flow-of-time][^src-most][^src-fence][^src-cvpe-2025]。

### 🎯 核心赌注

- **CB1 · 统一长程预测器：因果双分支 × 长期滞后核 × 可解释归因**〔A｜三目标全中〕：正面填中心缺口。[[e2-cstp|E²-CSTP]] 因果去混杂 + TIID 长期衰减 + 频率路由 + S-GRPO 反捷径 + 频带/滞后归因[^src-e2-cstp][^src-incident-guided-st-forecasting][^src-streasoner]。风险：集成复杂；E²-CSTP 薄增益警示"因果+多模态"难兑现，必须配 QW1 强机制与真基准[^src-e2-cstp]。
- **CB2 · 跨模态掩码预训练消歧器**〔A｜准确性+泛化〕：把 [[std-mae|STD-MAE]] 扩出"跨模态掩码"第三轴，预训练模型"用外生消歧内生"，直接操作化第一节的重构；STD-MAE 配方已验证、即插即用[^src-2312-00516-std-mae]。**性价比最高的核心赌注。**
- **CB3 · 检索式外生记忆 + 生成式长程概率预测**〔B｜准确性+不确定性〕：建"多模态外生时空记忆库"（[[rast|RAST]] 双维 + [[event-prior-augmentation|EPA]] 真实事件记忆 + 频域视角）[^src-rast][^src-uniextreme]，用 [[middir|MiDDiR]] 的解析性 score 偏置（免重训、专修罕见事件低密度区）注入物理时间扩散骨干（[[dyffusion|DYffusion]]/[[tedm|TEDM]]），[[swift|CRPS-AR]] 校准长程不确定性[^src-middir][^src-dyffusion][^src-tedm][^src-swift]。可借 [[nsdiff|NsDiff]] 让外生信息驱动异方差[^src-nsdiff]。

### 🌙 登月（高风险高回报）

- **MS1 · 气候态作长程外生驱动 + 数据同化**〔B｜泛化〕：把气候指数/季节背景当慢变外生驱动，用"生成式 emulator 即贝叶斯滤波器"（arXiv 2605.20028）同化含噪/部分外生观测，**统一交通-FM 与天气-FM 两条谱系**（全景页开放问题）。
- **MS2 · 智能体式外生信息主动获取 + mid-rollout 工具增强**〔B｜准确性+可解释〕：rollout 中用贝叶斯实验设计/不确定性趋势决定**查哪个外生信号**、mid-rollout 查外部 DB/API（To-Infinity 理论：工具增强解锁 SSM 长度泛化），轨迹级奖励解长程信用分配（AgentFlow）。CausalGame 显示前沿 agent 因果恢复全军覆没→风险很高。

### 🔧 横切基础设施（所有方向都依赖，应**最先做**）

- **EN1 · 基准（元方向，最高杠杆）**：拼接 MeteorPred MP-Bench（42 万气象场↔文本预警）+ [[e2-cstp|E²-CSTP]] 的 Terra/BjTT/GreenEarthNet + HR-Extreme-V2 + RealPDEBench[^src-e2-cstp][^src-uniextreme]。指标必须含：**lead-time 分辨的技能衰减、极端条件分层技能、每个外生模态的边际价值（模态消融）、归因忠实度**。没有它上面全部不可证伪——6 路挖掘独立确认此空白。
- **EN2 · 可扩展骨干**：分解线性（[[factost|FactoST]]）+ 改进 SSM（Mamba-3/DiM-TS）+ agent-token / [[stop|STOP ConAU]] 瓶颈**把"一个全局外生上下文"广播给所有节点**[^src-factost][^src-stop]。长程×大规模×多模态从第一天就是 OOM 问题。
- **EN3 · 归因工具箱**：把 FlashTrace（长程忠实归因）+ SAE 概念特征 + 因果中介移植到时空，回答"哪个外生因子、经多长滞后、驱动了哪个未来格点"，把可解释性从口号变成可测量。

## 六、推荐优先级与排序

> 领域的**绑定约束是"评测"**（6 路独立确认无基准）——所以**先做 EN1 基准 + QW1 滞后核插件**（基准是使能器，滞后核是最快的科学增量且能反压测基准）。
> → 再上 **CB2 跨模态掩码预训练**（操作化重构、即插即用、高杠杆）。
> → 待基准+插件+预训练就位，再做 **CB1 旗舰统一器**。
> → 生成式（CB3）与登月（MS1/MS2）作并行的高风险轨道。

一句话：**先把"怎么算赢"定下来（EN1），同时用 QW1 拿第一个干净的科学增量，主线压在 CB2→CB1，跨领域弹药（扩散引导/数据同化/智能体）作 B 路线储备。**

## 七、九条可迁移机制设计轴（速查）

| 轴 | 关键机制（论文标签） |
|---|---|
| **A 注入位置** | concat([[tats\|TaTS]] 零改架构) / 独立模态流+cross-attn(DiTS MM-DiT) / 条件重参数化归一化([[conformer\|ConFormer]] GLN) / 注意力引导信号([[aurora\|Aurora]]) / 解析性 score 偏置免重训([[middir\|MiDDiR]]) / 门控路由([[most\|MoST]] SNR) / 干净先验图([[pristi\|PriSTI]]) / 外部检索记忆([[rast\|RAST]]) |
| **B 长期影响★** | 参数化衰减核([[igstgnn\|TIID]]) / 多粒度滞后(METP) / 频率路由([[hyperd\|HyperD]]/[[dst-mamba\|DST-Mamba]]) / 状态依赖事件触发稀疏(Dynamic Sparsity) / 按相位检索([[rast\|GTR]]) |
| **C 可信度** | SNR 门控([[most\|MoST]]) / 观测接地 trust-aware / 有界误差融合(MUSE) / 反馈引导强度([[fence\|FENCE]]) / belief divergence vs conflict(DS 证据理论) / IB 过滤([[multimodal-semantic-understanding\|MindTS]]) |
| **D 因果** | 双分支去混杂([[e2-cstp\|E²-CSTP]]) / 因果解耦(CATAL/COGS) / 外生 regime 上 DRO([[stop\|STOP]]) / 反事实 ATE / 反捷径对比([[spatio-temporal-reasoning\|S-GRPO]]) |
| **E 可解释归因** | LLM 推理链([[event-driven-reasoning\|VoT]]) / 长程忠实归因(FlashTrace) / 机制数据归因(MDA) / SAE 概念特征(SARM) / 频带归因(VoT-AFF) |
| **F 骨干/效率** | 分解线性([[factost\|FactoST]]) / 改进 SSM(Mamba-3/DiM-TS) / agent-token 广播([[stop\|STOP ConAU]]) / TTT 快权重(tttLRM) / Kalman SSM(Gated KalmaNet) / 工具增强(To-Infinity) |
| **G 生成/不确定性** | 条件-去噪分离([[pristi\|PriSTI]]/[[ustd\|USTD]]) / 扩散步=物理时间([[dyffusion\|DYffusion]]/[[tedm\|TEDM]]) / CRPS-AR([[swift\|Swift]]) / 外生驱动异方差([[nsdiff\|NsDiff]]) / score Bayes 分解免重训(TSFlow-CPS) / emulator→贝叶斯滤波(2605.20028) |
| **H 数据/检索** | 多视角记忆([[urbandit\|UrbanDiT]]/UniFlow，频域记忆对周期外生最关键) / 真实事件记忆>可学习([[event-prior-augmentation\|EPA]]) / 检索"纠正后推理"([[event-driven-reasoning\|VoT-HIC]]) / OT 对齐静态外生+检索源城动态(CRAFT) / 图表结构化(FDV) |
| **I 评测★** | MP-Bench(气象场↔文本) / Terra/BjTT/GreenEarthNet([[e2-cstp\|E²-CSTP]]) / RealPDEBench(sim-to-real) / UnReason(飓风可视化+不确定性) |

## 未解决的问题与风险

- **外生信息并非总有益**：[[cvpe|CVPE]] 在弱相关数据集性能反降[^src-cvpe-2025]，[[tats|TaTS]] 增益取决于 CTR 共振水平[^src-language-in-the-flow-of-time]，[[middir|MiDDiR]] 引导强度强数据依赖[^src-middir]。缺一个原则性的"何时引入/引入多强"判据。
- **因果恢复很难**：CausalGame 中 16 个前沿 agent 在混杂下全部无法恢复因果结构；[[e2-cstp|E²-CSTP]] 的因果机制在真正多模态数据集上增益薄[^src-e2-cstp]——"长期因果影响机制"是否真能被可靠识别，存疑。
- **配对数据稀缺**：[[streasoner|STReasoner]] 主要靠合成数据训练，因真实 ST 数据集罕有配对的空间实体/依赖自然语言描述[^src-streasoner]；这是 EN1 基准必须解决的瓶颈。
- **LLM 路线的算力/延迟**：[[event-driven-reasoning|VoT]] 逐步 LLM 推理[^src-event-driven-ts-forecasting]、UrbanGPT 174s/传感器级延迟，大规模外生-LLM 时空预测的可行性未验证。
- **长上下文注意力稀释**：把长外生报告硬塞进 LLM，相关 token 注意力随上下文长度按 O(1/n) 衰减——长程外生推理须走检索/结构化记忆而非暴力长上下文[^src-event-driven-ts-forecasting]。

## 外部线索（2026 顶会汇编，尚未 ingest）

以下来自用户提供的《2026 AI/ML 顶会论文汇编》，**尚未进入 `raw/`**，作为未来 ingest 候选（按可迁移机制分组）：

- **文本→预测/可解释推理**：TESS（2603.12664，文本→时间原语中间瓶颈）、Bonsai（2504.03640，多模态证据组合式概率推理树）、Multimodal DeepResearcher（2506.02454，FDV 图表结构化）、FlashTrace（2602.01914，长程忠实归因）。
- **天气/气候外生**：MeteorPred（2508.06859，MP-Bench 气象场↔文本预警）、STCast（2509.25210，按气候态路由专家）、MoCast（物理引导降水临近预报）、WLA（2503.06623，天气潜空间）。
- **生成式引导/数据同化**：Generative Emulator as Bayesian Filter（2605.20028）、FloodDiffusion（2512.03520，时变控制事件注入）、GuideFlow（2511.18729，约束引导流匹配）、PhysicsCorrect（2507.02227，免训练长程 PDE 校正）。
- **长程骨干/世界模型**：tttLRM（2602.20160，TTT 长上下文）、Mamba-3（2603.15569）、SF-RSSM（快慢双支世界模型）、SparseWorld-TC（2511.22039，轨迹条件占据预测）。
- **智能体外生检索**：To-Infinity（2510.14826，工具增强解锁 SSM 长度泛化）、AgentFlow（2510.05592，轨迹级奖励）、HierSearch（2508.08088，本地+web 分层检索）、Shoot First（2510.20886，贝叶斯实验设计选择查询）。
- **图上多模态/因果**：MoMent（2502.19651，动态文本属性图链接预测）、HiFiNet（2511.12507，图频率分解）、在线影响因果建模（2505.19355，外生信号反事实 ATE）。
- **评测**：RealPDEBench（2601.01829）、UnReason（飓风可视化+不确定性）、STRIDE-QA（2511.17045，未来预测一致性指标）、GeoMMBench（2604.08896）。

## 关联页面

- [[spatiotemporal-mirage]] — 本页第一节重构的对象（消歧问题）
- [[spatio-temporal-foundation-model-landscape]] — 本页缺口分析的上游，自列三大未解问题
- [[multimodal-time-series-forecasting]] — 多模态时序预测的方法分类
- [[multimodal-semantic-understanding]] — 跨模态对齐/融合/冗余过滤的范式
- [[event-driven-reasoning]] — 外生文本→未来突变的 LLM 推理范式
- [[accident-aware-traffic-forecasting]] — 外生事件 ST 预测的问题形式化（5 挑战）
- [[heterogeneous-covariates]] — 异构外生协变量的统一接入
- [[retrieval-augmented-spatio-temporal-forecasting]] — 外部记忆扩展容量的范式
- [[e2-cstp]]、[[igstgnn]]、[[factost]]、[[visifold]]、[[swift]] — 四块拼图的最接近工作
- [[ctenet]] — 架构嵌入型 PINN，欧拉场 + ADR 方程，多模态外生在空气质量预测中的典范

## 引用源

[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
[^src-e2-cstp]: [[source-e2-cstp]]
[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
[^src-conformer]: [[source-conformer]]
[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
[^src-factost]: [[source-factost]]
[^src-visifold]: [[source-visifold]]
[^src-swift]: [[source-swift]]
[^src-most]: [[source-most]]
[^src-aurora]: [[source-aurora]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-dst-mamba]: [[source-dst-mamba]]
[^src-middir]: [[source-middir]]
[^src-dyffusion]: [[source-dyffusion]]
[^src-tedm]: [[source-tedm]]
[^src-pristi]: [[source-pristi]]
[^src-fence]: [[source-fence]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]
[^src-unica]: [[source-unica]]
[^src-streasoner]: [[source-streasoner]]
[^src-uniextreme]: [[source-uniextreme]]
[^src-rast]: [[source-rast]]
[^src-stop]: [[source-stop]]
[^src-nsdiff]: [[source-nsdiff]]
[^src-ctenet]: [[source-ctenet]]
