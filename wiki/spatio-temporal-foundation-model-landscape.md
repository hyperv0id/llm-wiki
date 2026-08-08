---
title: "时空基础模型全景"
type: analysis
tags:
  - foundation-model
  - spatio-temporal
  - zero-shot
  - scaling-laws
  - architecture-design
  - landscape-analysis
  - paradigm-comparison
  - problem-evolution
created: 2026-06-01
last_updated: 2026-08-08
source_count: 12
confidence: high
status: active
---

# 时空基础模型全景

## 溯源地图

```
         [问题0] 每个城市单独训练，换城市就要重来
             |
             v
         [问题1] 怎么跨城市零样本预测？
         /       |        \
        v        v         v
    GPT-ST     UniST    UrbanGPT
    (先训再搬)  (一训全用)  (LLM的世界知识)
        |        |          |
        v        v          v
    [问题2] 跨了城市，但空间信息反而拖后腿（负迁移）
             |
             v
    +---------------------------+
    | 三种应对策略              |
    | A. 消除空间统计依赖       |  OpenCity (Instance Norm)
    | B. 用 prompt 注入空间先验 |  UniST/UrbanDiT/UniFlow (Memory Pool)
    | C. 干脆把时空拆开         |  FactoST (UTP+STA)
    +---------------------------+
             |
             v
    [问题3] 拆开之后，空间信息什么时候注入？怎么注入？
             |
             v
    FactoST: 延迟到适配阶段 (STA)    UrbanFM: 在注意力内部拆 (分解注意力)
             |                           |
             v                           v
    [问题4] 单任务不够，多任务/多模态怎么办？
             |
             v
    UrbanDiT: 掩码=任务 (5任务)   MoST: SNR门控选模态   BIGCity: ST-unit统一轨迹+交通
             |                        |                       |
             v                        v                       v
    [问题5] 终局：解耦预训练 + 极简架构 + 多模态原生 + 生成式概率预测
```

---

## 问题 0：每个城市要单独训练，换城市就要重来

时空预测的传统做法是：拿到一个城市的数据，从头训一个模型，调参，部署。换一个城市，全部重来。这条路走了很多年，从 [[stgcn|STGCN]] 到 [[gwnet|GWNet]] 到 [[mtgnn|MTGNN]]，模型越做越精，但每个模型都绑死在一个数据集上[^src-most]。

问题很明显——成本高、泛化差。部署一个新城市需要收集数据、训练模型、调超参，动辄几周[^src-most]。更致命的是，模型学会了北京路网的拓扑，换个上海就全废了，因为上海的传感器位置、道路连接、流量分布完全不同[^src-most]。

所以第一个问题就自然浮出来了：**能不能训一个模型，直接用到没见过的城市？**

## 问题 1：怎么跨城市零样本预测？

三条路几乎同时出现，每条路看到的问题都一样，但应对方式完全不同。

### 路 A：先训再搬——GPT-ST

[[gpt-st|GPT-ST]] (NeurIPS 2023) 是第一个尝试"先训练、再迁移"的时空模型。思路很直：在目标城市的图上做 MAE 预训练（随机遮住一些传感器数据，让模型学着猜回来），学到的表征再喂给下游预测器[^src-gpt-st]。

它证明了一件事：预训练确实有用，不需要从零开始。但它留下一个硬伤——**每个城市要单独预训练一次**。在北京预训练的表征不能直接拿去上海用，因为预训练阶段已经把北京的图结构焊死在模型里了[^src-gpt-st]。所以它不是真正的基础模型，更像是一个"加速器"——让你每次训练更快，但还是要训练。

### 路 B：一训全用——UniST

[[unist|UniST]] (KDD 2024) 说：既然每个城市单独训不行，那我就把 20 多个数据集混在一起训一个大模型，然后冻结 backbone，只调 prompt[^src-unist]。

关键创新是 **knowledge-guided prompt learning**：从输入数据中提取 4 个"知识点"（空间近邻、空间层级、时间近邻、时间周期），用它们去查一个 learnable memory pool，查出来的结果作为 prompt 注入 Transformer。每个城市的数据虽然格式不同，但背后的"早高峰模式""功能区效应"是类似的——memory pool 存的就是这些跨城市的共享模式[^src-unist]。

它确实做到了 zero-shot 跨城市。但问题来了：**prompt 从哪来？** 目标城市的数据你还没见过，你怎么提取"空间近邻"？答案是——你需要目标城市的图结构信息。UniST 的 prompt 依赖输入数据的空间特征，一旦目标城市没有足够的历史数据，prompt 就退化了[^src-unist]。

### 路 C：LLM 替你理解城市——UrbanGPT

[[urbangpt|UrbanGPT]] (KDD 2024) 走了一条完全不同的路：别自己从零学城市知识了，LLM 已经读过几百万篇城市相关的文本，它*已经知道*商业区和住宅区的交通模式有什么区别[^src-urbangpt]。

做法是把时间序列编码成特殊 token 喂给 Vicuna-7B，再用自然语言描述（"这是 Staten Island，附近有公共安全、教育、住宅类 POI"）让 LLM 理解空间语义。LLM 输出的不是数字，而是一个 4096 维的"语义推理向量"，再通过回归层翻译成预测值[^src-urbangpt]。

它的致命缺陷在 ablation 里暴露无遗：**移除回归层（让 LLM 直接输出数字）是最严重的退化**[^src-urbangpt]。LLM 擅长推理，不擅长算数——你让一个哲学家去做精算，结果一塌糊涂。加上 7B 参数导致推理 174 秒/传感器，实际部署几乎不可行。

**三条路的共同遗留问题**：虽然都能跨城市了，但目标城市的空间信息（图结构、POI 分布、传感器坐标）仍然是必需输入。更深层的问题是——这些空间信息在新城市上到底帮了你，还是害了你？

## 问题 2：空间信息反而拖后腿——负迁移

[[factost|FactoST]] 发现了一个反直觉的事实：**把空间信息写进预训练，跨城市反而变差**[^src-factost]。

怎么发现的？FactoST 在零样本上对比了 [[opencity|OpenCity]]（联合时空预训练）和 [[timesfm|TimesFM]]（纯时间序列预训练，完全不管空间）。结果 TimesFM 赢了。一个根本不看地图的模型，比一个精心编码了图拓扑的模型，在没见过的城市上预测更准[^src-factost]。

为什么？因为预训练时学到的空间编码绑死了源域的图结构。北京的环路拓扑和上海的放射状路网差异巨大——你在北京学到的"邻居模式"在上海完全不适用，甚至会误导。这就是**负迁移**：本来想帮忙的信息，到了新场景反而成了干扰[^src-factost]。

三种应对策略应运而生：

### 应对 A：消除空间统计依赖——OpenCity

[[opencity|OpenCity]] 说：既然源域的空间统计在新城市上会干扰预测，那我就把所有城市的统计归一化掉[^src-opencity]。

做法是 Instance Normalization——每个传感器用自己的均值和标准差归一化，预测完再反归一化。这样模型看到的永远是标准化后的值，不管北京还是上海，"0.3"代表的都是"比平时高 30%"[^src-opencity]。同时用 Laplacian 特征向量编码图拓扑（只需要邻接矩阵，不需要历史数据），TimeShift Transformer 显式建模日/周周期。

它做到了真正的 zero-shot（不需要目标城市的任何训练数据）。但代价是：**你仍然需要新城市的邻接矩阵**。一个没有路网数据的新城市，OpenCity 也无能为力[^src-opencity]。

### 应对 B：用 prompt 注入空间先验——UniST / UrbanDiT / UniFlow

这条路线的思路是：预训练学到的通用表征不够精确，需要用 prompt/memory 机制把城市特有的知识注入[^src-unist][^src-urbandit]。

如果你把 UniST 的 4 个 prompt 去掉，换成 UrbanDiT 的 3 个 memory pool（时域/频域/空域），再换成 UniFlow 的 4 组 ST-MRA（时域/频域/时空/频空），本质上是一回事——**存一批跨城市的共享模式，推理时查出来当 prompt 注入**。区别只是存了多少种、怎么查的。

这条路线在 few-shot 上很强（少量数据就能调好 prompt），但在 zero-shot 上受限于 prompt 质量——你还没见过目标城市，memory pool 里可能没有合适的 pattern 可查[^src-unist]。

### 应对 C：干脆把时空拆开——FactoST

[[factost|FactoST]] 做了一个最激进的决定：**预训练阶段完全不碰空间**[^src-factost]。

它的 Pattern Factorization Hypothesis 说：时空数据里，时间模式是跨城市通用的（早晚高峰哪里都有），但空间模式是每个城市独有的（路网拓扑不可能一样）。所以，先在 11B+ 时间点上只训时间动态（UTP 阶段），完全不管图结构；训好了，再用一个轻量适配器（STA）把空间信息注入[^src-factost]。

结果是：零样本上一致优于所有联合模型，而且算力需求从 O(N²) 降到 O(N)——因为时间部分是独立处理的，空间适配只加线性开销[^src-factost]。

但这个解法又留下了新问题：**空间信息什么时候注入？怎么注入？**

## 问题 3：拆开之后，空间信息什么时候注入、怎么注入？

两个模型给出了两个截然不同的答案。

### 答案 1：延迟到适配阶段——FactoST

FactoST 的答案是"晚点再说"。预训练纯搞时间，到适配阶段才用 STA 模块注入空间[^src-factost]。STA 里有 4 个组件：STMF 给基础时空上下文，STF 动态加权三种亲和度（空间兼容性、模式对齐、滞后因果），DSPA 用低秩 prompt token 对齐源域和目标域的分布，CMR 防遗忘。

好处是负迁移风险极低——预训练时根本没有空间编码，所以不可能冲突。坏处是 STA 里的 node embedding 是 transductive 的——每个传感器要单独学一个嵌入向量，新节点加入时需要重新训练[^src-factost]。这不算真正的 inductive，更像是"先训时间，再微调空间"。

### 答案 2：在注意力内部拆——UrbanFM

[[urbanfm|UrbanFM]] 说：不需要分两个阶段，在一个模型里把注意力拆成两半就行[^src-urbanfm]。

先做时间注意力（同一传感器内部，不同时间步之间的 attention），再做空间注意力（同一时间步内，不同传感器之间的 attention）。不做时空联合的 O((NT)²) 注意力，而是做 O(NT²) + O(N²T) 的分解注意力。配上 ST-RoPE（时间用相对时间距离、空间用相对空间序编码位置），RevIN 处理非平稳性，MiniST 用 KD-Tree 把异构传感器统一成可学习 token[^src-urbanfm]。

如果你把 FactoST 的"两阶段"拉到一个模型里，UTP 变成时间注意力，STA 变成空间注意力——你得到的几乎就是 UrbanFM。两派从不同的路走到了同一个地方：**时间和空间应该在不同的子模块里分别处理**。

唯一不同的是注入时机：FactoST 是"先训时间，后调空间"的松耦合，UrbanFM 是"同时训但分头做注意力"的紧耦合。前者更安全（零样本更好），后者更灵活（zero-shot 同时也支持 imputation）[^src-factost][^src-urbanfm]。

## 问题 4：单任务不够，多任务/多模态怎么办？

到这一步，跨城市零样本基本解决了。但实际部署要的不是"只能预测"——你可能同时需要预测、插值、填补缺失值、外推新区域。而且数据也不只是数字，还有卫星图、POI 文本、轨迹序列。

### 多任务：用掩码当任务开关

[[urbandit|UrbanDiT]] 的做法最优雅——5 种任务不需要 5 个输出头，只需要 5 种掩码策略[^src-urbandit]。掩码未来 = 前向预测，掩码过去 = 后向预测，掩码中间时间点 = 时间插值，掩码未知空间区域 = 空间外推，随机掩码 = 缺失值填补。同一个扩散 Transformer，任务不同只是"遮住的地方不同"。加上 rectified flow 做 25× 推理加速，5 个任务一个模型搞定[^src-urbandit]。

[[bigcity|BIGCity]] 把任务类型推得更远——它统一了两种以前完全分开的数据：个体轨迹（你打车从 A 到 B 经过的路线）和群体交通状态（某条路现在的平均车速）。核心发明是 ST-unit：一个三元组（道路段静态特征 + 动态交通状态 + 时间戳），不管是轨迹还是交通状态，都是 ST-unit 的序列[^src-bigcity]。模型看到的不再是"轨迹"或"交通"，而是"一串道路段的序列"——它根本不知道自己在处理哪种数据。结果：8 个任务全部 SOTA，跨模态训练比同模态训练收益更大[^src-bigcity]。

### 多模态：谁来决定看什么？

[[most|MoST]] 是第一个原生多模态的时空基础模型——卫星图像、POI 文本、位置坐标、时间序列四种输入[^src-most]。问题来了：不是每个城市都有卫星图，POI 数据质量也参差不齐。如果无脑全用，垃圾模态会拖垮好模态。

MoST 的解法是 SNR 门控：估计每个模态的"信噪比"——高质量模态放行，低质量模态用 Gumbel-Sigmoid 门关掉[^src-most]。这像收音机调频：信号强的时候听，信号弱的时候静音，不勉强。

## 问题 5：终局——时空基础模型最终会长什么样？

回看整条线，五个问题催生了一个收敛趋势：

1. 每个城市单独训练 → 要跨城市零样本 → 空间信息带来负迁移 → 要时空解耦 → 解耦后空间怎么注入 → 多任务多模态怎么统一

三个趋势正在汇聚：

**趋势 1：时空解耦成为共识。** FactoST 在阶段层面拆，UrbanFM 在注意力层面拆，但核心洞察一样——时间和空间应该分别处理。这不是技术偏好，是物理现实：时间模式跨城市通用，空间模式每个城市独有，硬绑在一起只会互相干扰[^src-factost][^src-urbanfm]。

**趋势 2：适配机制在简化。** UniST 需要 prompt-tuning，UrbanFM 直接 zero-shot——中间省掉了一步。这和 Sutton 的 "Bitter Lesson" 一致：当你数据够多、架构够通用，手工设计的适配机制就会被大规模学习取代[^src-urbanfm]。

**趋势 3：单模态走向多模态。** MoST 证明了多模态输入比单模态好，BIGCity 证明了跨数据类型比同类型好[^src-most][^src-bigcity]。下一代 STFM 原生支持多模态几乎是确定的。

### 预测

最终的时空基础模型大概是这四样东西拼起来：

- **FactoST 的解耦框架**——预训时间、延迟注入空间
- **UrbanFM 的极简 scaling**——少搞先验、多堆数据
- **MoST 的多模态**——SNR 门控选模态
- **UrbanDiT 的扩散概率预测**——不仅能预测，还能告诉你"我有多不确定"

---

## 压缩总览图

```
+--------+--------+--------+--------+--------+--------+
| 问题 0 | 问题 1 | 问题 2 | 问题 3 | 问题 4 | 问题 5 |
| 换城市  | 怎么跨 | 空间拖 | 拆开后 | 多任务  | 终局   |
| 要重来  | 城市？  | 后腿？  | 怎注入？| 多模态？| 形态？ |
+--------+--------+--------+--------+--------+--------+
    |        |        |        |        |        |
    v        v        v        v        v        v
 单城模型  3条路   3种应对   2种答案   3种统一   4拼一
 STGCN    GPT-ST  OpenCity  FactoST  UrbanDiT  解耦+
 GWNet    UniST   (消除依赖) (延迟注入) (掩码=任务) 极简+
 MTGNN    UrbanGPT UniST等  UrbanFM  BIGCity   多模态+
          (先训搬) (prompt)  (内拆)   MoST      扩散
                  (LLM知识)                    概率
                                                             
 关键转折: 问题2→3  空间信息从"帮助"变成"干扰"
                     → 从"怎么用空间"变成"什么时候用空间"
```

---

## 未解决的问题

- **开放世界空间**：所有 STFM 仍需目标域的空间信息。真正的零样本应无需任何拓扑知识。FactoST 的 transductive node embeddings 是一个未解的局限[^src-factost]。
- **时空幻象**：[[spatiotemporal-mirage|短输入窗口导致预测困境]]，[[std-mae|STD-MAE]] 通过 864 步预训练缓解，但推理时仍受限于下游模型的输入长度[^src-2312-00516-std-mae]。
- **外生事件**：[[conformer|ConFormer]] 和 [[igstgnn|IGSTGNN]] 证明事故对交通有剧烈影响，但当前 STFM 均未建模外生事件。[[vot|VoT]] 和 [[timecap|TimeCAP]] 尝试用 LLM 提取事件信号，但尚未与 STFM 架构集成。系统的研究路线（统一架构论点 + 分层方向）见 multimodal-exogenous-guided-long-term-st-forecasting。
- **不确定性量化**：STFM 的点预测已超越 full-shot 专家，但概率预测仍是空白。[[ustd|USTD]] 用扩散模型做概率预测但非基础模型范式。FactoST 的分位数预测头是当前最接近的方案[^src-factost]。
- **跨域泛化边界**：当前 STFM 泛化主要在交通领域验证。天气/地球系统领域形成独立 FM 谱系（NVIDIA Atlas、WIND、ESFM），两者能否统一仍未知。
- **评估标准化**：EvalST（12 数据集、22 baseline）是迄今最系统的城市 ST 基准，但尚未被社区广泛采用。STFM 领域需要一个等价于 GIFT-Eval 的统一基准。

## 综述文献索引

| 综述 | 年份 | 来源 | 特色 |
|------|------|------|------|
| Fang et al. | 2026 | IEEE TKDE (arXiv:2506.01364) | 最全面，Pipeline 视角 |
| Liang et al. | 2025 | KDD Tutorial (arXiv:2503.13502) | sensing+management+mining 全工作流 |
| Goodge et al. | 2025 | arXiv:2501.09045 | 形式化定义，批判性评估 6 个 STFM |
| Mao et al. | 2025 | ACM Computing Surveys | Transformer→FM 演化路线 |
| Jin et al. | 2024 | arXiv:2310.10196 | LM4TS + LM4STD 分类 |

## 深层洞见

整条演化线背后，真正在发生的变化不是技术迭代，而是一次**对"空间"的认知降级**：

- 2023 年，GPT-ST 相信空间是必须精心编码的核心信号
- 2024 年，UniST/OpenCity 相信空间有用但需要适配
- 2025 年，FactoST 证明空间在预训练阶段是噪声，应该推迟到适配阶段
- 2026 年，UrbanFM 认为空间只需要在注意力里分一块出来，不需要额外先验

空间从"主角"变成了"配角"，最后变成了"可以延迟注入的附件"。这个降级的根源是**负迁移**——越是精心编码的领域知识，在跨域迁移时越容易变成负担。Sutton 的 Bitter Lesson 在时空领域重演了：通用方法 + 大规模数据 > 领域知识 + 小规模数据。

下一步最可能的方向：**空间信息完全从模型架构中移除，变成纯输入特征**。如果 MiniST/KD-Tree 隐式编码的思路被验证，未来的 STFM 可能根本不需要邻接矩阵、Laplacian 特征向量、GCN——只要给模型传感器的坐标和坐标附近的文本描述，它就能自己学会空间关系。

---

## 引用源

[^src-most]: [[source-most]]
[^src-urbanfm]: [[source-urbanfm]]
[^src-factost]: [[source-factost]]
[^src-urbandit]: [[source-urbandit]]
[^src-unist]: [[source-unist]]
[^src-uniflow]: [[source-uniflow]]
[^src-urbangpt]: [[source-urbangpt]]
[^src-opencity]: [[source-opencity]]
[^src-bigcity]: [[source-bigcity]]
[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
[^src-urbanpg]: [[source-urbanpg]]
[^src-aurora]: [[source-aurora]]
