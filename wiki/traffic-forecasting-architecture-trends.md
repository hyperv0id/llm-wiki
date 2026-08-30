---
title: "交通预测架构趋势"
type: analysis
tags:
  - traffic-forecasting
  - architecture-evolution
  - landscape-analysis
created: 2026-08-26
last_updated: 2026-08-26
source_count: 68
confidence: high
status: active
---

# 交通预测架构趋势

> 本页回答一个问题：**近年交通预测的架构为什么从"手工设计的图模型"演变成"预训练+适配+生成式"的形态**。结构按问题组织，不是按时间排列——每节先提出设计矛盾，再用论文证据说明不同方案如何回应它。所有论断按论文口径归因（论文自述 vs 实证 vs 课程推断），不把时间先后说成因果。

## 起点：一个图建模问题

交通预测的传统任务定义是：给定传感器网络的历史观测，预测未来若干步的流量/速度/需求。统计方法（ARIMA、VAR、SVR）无法捕获非线性空间相关性，这是深度学习进入该领域的前提[^src-hyperd-hybrid-periodicity-decoupling]。早期深度模型的共同起点是把路网当作图：节点是传感器，边是路网连接。

但"把路网当作图"之后，设计空间里立即出现一连串矛盾，后续所有架构变迁都是对这些矛盾的选择。

---

## 矛盾一：图先验 vs 数据驱动——预定义图可信吗？

**问题**：空间依赖关系是否等同于物理连接？若不等同，该信谁的？

早期模型把物理路网当作先验直接使用。[[dcrnn|DCRNN]]（ICLR 2018）把交通传播形式化为有向图上的扩散过程：沿出边/入边做截断扩散卷积（K=3），用扩散卷积 GRU 编码历史、Seq2Seq + scheduled sampling 解码未来[^src-dcrnn]。论文报告三个消融结论：有向图优于无向图、双向扩散优于单向、扩散优于无图（NoConv 退化到普通 RNN）[^src-dcrnn]。但 DCRNN 的邻接矩阵由距离阈值高斯核预定义且静态，论文自述其局限是无法适应新道路、封路或非物理的相关性[^src-dcrnn]。[[stgcn|STGCN]]（IJCAI 2018）走谱域 Chebyshev 图卷积 + 门控 1D 因果卷积的"时间→空间→时间"三明治块，论文报告训练速度比 RNN 系快一个数量级（PeMSD7-M 上 272s vs GCGRU 3825s），并声称图卷积一致优于非图方法[^src-stgcn]。它同样依赖预定义邻接矩阵，且假设图结构时不变——论文自述特殊事件（事故封路）时不适用[^src-stgcn]。

回应"预定义图不可靠"的第一种机制是注意力：[[source-astgcn|ASTGCN]]（AAAI 2019）让空间注意力矩阵与邻接矩阵逐元素相乘、时间注意力动态加权时间片，论文报告注意力权重与地理邻近性对齐，优势随预测长度增大而扩大[^src-astgcn]。

真正的范式转折是 [[gwnet|GWNet]]（IJCAI 2019）：自适应邻接矩阵 $\tilde{A}_{adp} = \text{SoftMax}(\text{ReLU}(E_1 E_2^\top))$ 从随机初始化的节点嵌入端到端学出隐藏空间依赖[^src-gwnet]。论文报告消融：仅自适应矩阵（MAE 3.10）几乎追平前向后向扩散（3.08），两者结合最优（3.04）[^src-gwnet]。GWNet 同时用膨胀因果卷积获得指数级感受野，非自回归一步输出全部未来步[^src-gwnet]。论文自述 GWNet 72% 的参数在节点嵌入中——这成为后续 [[ragc|RAGC]] 做嵌入正则化的动机[^src-ragc-efficient-traffic-forecasting]。

[[mtgnn|MTGNN]]（KDD 2020）把图学习推向"完全不用预定义图"：图学习层 + mix-hop 传播 + 扩张初始层联合训练，论文报告在 METR-LA/PEMS-BAY 上与使用物理拓扑的 STGNN 持平，但学到的邻居"分布更远但位于相同道路"，比预定义近邻更能提前预示极端交通[^src-mtgnn]。论文自述其局限：小图（Exchange 8 节点）上失效、图学习层 O(N²) 推理复杂度[^src-mtgnn]。

**小结**：空间建模从"预定义物理图"（DCRNN/STGCN）走向"数据驱动图"（GWNet 自适应邻接、MTGNN 图学习层）。但所有早期模型仍是单数据集专用：无跨数据集预训练、无概率输出、无事件感知[^src-stgcn][^src-dcrnn]。这一段证据只说明"自适应图在报告的数据集上不劣于物理图"，不构成"图先验已死"的结论——后文 [[opencity|OpenCity]]、[[gamma-net|GAMMA-Net]] 仍在使用预定义图结构[^src-opencity][^src-gamma-net]。

---

## 矛盾二：容量 vs 效率——注意力值得吗？

**问题**：Transformer 的 O(T²) 注意力换来多少精度？当线性基线把它打穿后，什么设计才让注意力重新值钱？

### 线性基线的冲击

[[source-zeng-2022-are-transformers-effective|Zeng et al. 2022]] 提出尖锐质疑：自注意力是排列不变的，天然丢失时序顺序信息；单层线性模型（DLinear 等）在九个基准上以 20%–50% 的优势击败所有 Transformer 变体[^src-zeng-2022-are-transformers-effective]。论文报告的证据包括：Transformer 随回看窗口从 96 增至 720 性能停滞甚至变差，而线性模型持续改善；打乱输入顺序几乎不影响 Transformer（降 0.09%–1.98%）但线性模型下降 27%–46%；将 Informer 逐步简化到线性层反而提升性能（Exchange MSE 0.847→0.084）[^src-zeng-2022-are-transformers-effective]。论文给出的出路是"引入时序归纳偏置而非依赖通用自注意力"，FEDformer 的频域处理是论文列举的竞争性案例[^src-zeng-2022-are-transformers-effective]。

### 让注意力重新值钱：patch、倒置、周期

[[patchtst|PatchTST]]（ICLR 2023）证明正确设计的 Transformer 可以击败线性基线：patch 化（局部语义 + 注意力 token 数降 S² 倍）+ 通道独立（共享权重），论文报告在 8 个数据集上相对最佳 Transformer 基线 MSE 降 21.0%，且是唯一随回看窗口增大持续改善的 Transformer[^src-patchtst]。[[itransformer|iTransformer]]（ICLR 2024）反转维度：每个变量的整条序列嵌入为一个 variate token，注意力建模变量相关性、FFN 建模序列表示，论文报告在 Traffic（862 变量）等场景显著占优，且反转框架可套在任何 Transformer 变体上（平均 MSE 提升 16.8%–38.9%）[^src-itransformer]。

频域/周期归纳偏置是另一条修正路线。[[autoformer|Autoformer]]（NeurIPS 2021）用自相关机制（FFT 求周期 + 按延迟滚动对齐）替代点级注意力，复杂度 O(L log L)，并把序列分解（趋势/季节）内嵌为模块[^src-autoformer]。[[timesnet|TimesNet]]（ICLR 2023）用 FFT 发现 top-k 周期，把 1D 序列重塑为 2D 张量（周期内/周期间变化）用 2D 卷积处理，论文报告在长时预测、插补、分类、异常检测五个任务上持续 SOTA[^src-timesnet]。

交通领域同步引入注意力与 MoE：[[testam|TESTAM]]（ICLR 2024）是论文自称的首个 MoE 时空注意力模型——identity/可学习静态图/空间注意力三类异构专家经记忆增强门控路由，仅 224K 参数即达 METR-LA/PEMS-BAY/EXPY-TKY SOTA，其 time-enhanced attention 从历史直接注意目标时间步，消除自回归误差累积[^src-testam]。[[staeformer|STAEformer]] 是交通 Transformer 的常用基线，被 ConFormer 等作为对比对象[^src-staeformer][^src-conformer]。

**小结**：这一轮证据的教训是"先验 > 容量"——纯注意力不敌线性基线，但注入 patch/倒置/周期先验后重新占优。二次复杂度没有在这里被解决，只是被绕开，成为下一节的导火索。

---

## 矛盾三：生成式输出 vs 确定性精度——分布值得吗？

**问题**：预测该输出一个点，还是输出一个分布？分布能换来什么，又付出什么代价？

确定性模型只输出点估计，无法量化不确定性[^src-2401-08119-specstg]。[[diffstg|DiffSTG]]（AAAI 2023）论文自称首次把 DDPM 推广到时空图：mask 未来作为条件信号、UGnet（门控 TCN + GCN）去噪、非自回归一次生成全窗口，论文报告推理比 TimeGrad 快约 40 倍，CRPS 降 5.6%–14.3%[^src-diffstg]。但论文报告确定性精度落后最佳 STGNN 约 5%–10%——ELBO 优化不会产生和直接 RMSE 优化同样锐利的后验[^src-diffstg]。

[[ustd|USTD]]（SIGSPATIAL 2024）用"预训练时空编码器 + 任务专用去噪解码器"的解耦训练打破"扩散 STG 打不过确定性基线"的现状：graph sampling（80% 节点子采样）+ MAE 式掩码预训练，TGA/SGA 分别在时间/空间轴做交叉注意力，论文报告 PEMS-BAY 上 CRPS 降 12%，首次在预测任务上超越确定性基线[^src-ustd]。[[specstg|SpecSTG]]（2024）把扩散搬到图谱域：生成未来序列的图傅里叶表示而非原始序列，快速谱图卷积把复杂度从 O(N²) 降到 O(N)，论文报告点估计 RMSE 最高降 8%、训练速度是 GCRDD 的 3.33 倍[^src-2401-08119-specstg]。[[tedm|TEDM]]（ICLR 2026）把 EDM 设计空间迁移到时序并把扩散时间轴与物理时间轴对齐，采样复杂度从 O(SH) 降到 O(H)；用从数据经验估计的噪声/尺度 schedule 替代人工预设，论文报告 ETTh2 上相对 EDM 提升最高 85% MSE[^src-tedm]。

流匹配路线解决扩散的采样成本。[[tsflow|TSFlow]]（ICLR 2025）论文自称首个把条件流匹配用于概率时序预测：GP 先验（SE/OU/周期核）替代各向同性高斯、最优传输耦合缩短概率路径，论文报告 6/8 数据集 CRPS SOTA[^src-tsflow]。[[flowts|FlowTS]]（2025）用 rectified flow 的直线输运做时序生成，论文报告仅 30 采样步即超越 Diffusion-TS 的 200 步，Solar 预测 MSE 相对最优基线降 43.2%[^src-flowts]。[[sundial|Sundial]]（ICML 2025）把流匹配做成时序基础模型的训练目标：TimeFlow Loss 让自回归 Transformer 直接在连续值域学习每个 patch token 的预测分布，论文报告 GIFT-Eval 上 MASE 排名第 1，且生成式目标缓解了 MSE 训练导致的过平滑预测[^src-sundial]。

交通领域出现面向任务的生成式方案：[[craft|CRAFT]]（NeurIPS 2025）用检索增强扩散做零样本跨城交通流生成——地理特征对齐（最优传输）解决域偏移、检索条件增强补充时间动态，论文报告在四个共享单车数据集上比基线平均提升 59.7%[^src-craft]。[[loft|LOFT]]（KDD 2026）用低秩先验 + 一致性轨迹目标做高稀疏交通插补，论文报告推理仅 2 NFE 且保持精度[^src-loft]。

**小结**：扩散把分布输出带进时空预测，代价是确定性精度（DiffSTG 自述落后 5%–10%）；USTD 用预训练弥合；流匹配把采样步数从 50–200 压到 20–30；生成式目标还反向服务于点预测（Sundial）与插补（LOFT）。

---

## 矛盾四：专用 vs 泛化——单城模型能跨城吗？

**问题**：一个在纽约训好的模型，搬到北京还能用吗？要跨城，先想清楚"什么该共享、什么该私有"。

### 基础模型：统一预训练 + 适配

[[urbandit|UrbanDiT]]（NeurIPS 2025）把扩散 Transformer 引入时空基础模型：DiT + rectified flow 统一 grid/graph 两种数据，用"掩码=任务"的策略把前向预测、插值、外推、插补统一为重建任务，三个 memory pool（时域/频域/空域）+ 任务 prompt 注入，论文报告零样本超越多数有训练数据的基线[^src-urbandit]。

[[opencity|OpenCity]]（2024）主打纯数值零样本：Instance Norm 消除训练集统计依赖 + patch embedding + TimeShift Transformer（周期跨注意力 + 动态自注意力）+ Laplacian 特征向量空间编码 + GCN 聚合，在 21 个数据集（1.51 亿观测）上预训练，论文报告零样本在 4/6 测试集超越 full-shot 基线，推理 <3s[^src-opencity]。论文自述其局限：仍需预定义邻接矩阵、单模态[^src-opencity]。

### 空间信息的降级：解耦与负迁移

[[factost|FactoST]]（NeurIPS 2025/arXiv 2026）提出 Pattern Factorization Hypothesis：时空数据中时间模式跨域通用、空间模式每城独有，联合预训练时空模式反而引入负迁移。论文据此分两阶段：先做纯时间预训练（UTP，11B+ 时间点、8 域、encoder-only），再用轻量适配器（STA）注入空间[^src-factost]。论文报告零样本超越 TimesFM、Moirai、OpenCity、UniST，10% 标注数据即接近 full-shot 性能；消融显示随机序列掩码最关键（移除 +17.7% MAE）[^src-factost]。论文自述其局限：STA 节点嵌入仍是 transductive 的[^src-factost]。

[[urbanfm|UrbanFM]]（2026）以 scaling 为中心：WorldST 数据（100+ 城市、10 亿+ 数据点，论文报告是 UniST/OpenCity/BigCity 的 33–145 倍）、MiniST（KD-Tree 聚类统一异构传感器）、极简分解注意力（时间注意力 + 空间注意力分开做）+ ST-RoPE + RevIN，论文报告零样本 MAPE 超越现有时空基础模型 39%–70.2%，未训练 imputation 也能在 PEMS 填补任务上最优[^src-urbanfm]。论文报告预训练数据量指数扩展下 MAPE 幂律衰减、未见饱和[^src-urbanfm]。

[[stunet|STUNet]]（KDD 2026）从显式结构 token 化切入跨网络泛化：把邻接矩阵切成空间 patch 冻结为 spatial tokens，时间 tokens 经 query-aggregate attention 查询上下游，两阶段训练（tokenizer 预训练 + 冻结空间 tokenizer 微调）保证空间表示不被时间信号回写[^src-stunet]。论文报告在互不重叠的 LargeST 子网络上跨网络零样本全面最优（SD→GBA MAE 34.46 vs PatchSTG 37.98）[^src-stunet]。

### 多模态与多任务

[[most|MoST]]（KDD 2026）论文自称首个多模态时空基础模型：卫星图（ResNet50）、POI 文本（BERT）、坐标、时序四种输入，SNR 门控（Gumbel-Sigmoid）自适应屏蔽低信噪比模态，多模态引导空间专家建模局部交互，论文报告零样本超越多数 full-shot 端到端模型与 OpenCity[^src-most]。论文自述依赖预训练模态编码器，且卫星/POI 数据并非所有城市可得[^src-most]。

[[bigcity|BIGCity]]（2024）用 ST-unit 三元组（静态路网特征 + 动态交通状态 + 时间戳）统一个体轨迹与群体交通状态两类数据，GPT-2 + LoRA + 任务导向 prompt 覆盖 8 个异构任务，论文报告跨模态多任务训练的正迁移强于同模态[^src-bigcity]。

[[urbanpg|UrbanPG]]（AAAI 2026）解耦"个性化上下文提示 + 通用骨干"，用线性时空上下文注意力（STCA）把复杂度降至 O(N·d²)，一个框架同时支持大规模预测、few-shot 泛化与持续学习（冻结骨干仅扩展 prompt）[^src-urbanpg]。[[std-plm|STD-PLM]]（AAAI 2025）用 GPT-2 前 3 层做 PLM backbone，显式空间/时间 tokenizer + sandglass 注意力（M<N 区域 token 压缩），论文报告 few-shot 5% 数据即匹配全量 LSTM[^src-std-plm]。[[moirai-moe|Moirai-MoE]]（ICML 2025）把稀疏 MoE 引入时序基础模型：token 级专家专业化（k-means 门控）替代频率级分组，论文报告 11M 激活参数超越 dense Moirai-S 17%[^src-moirai-moe]。

两份综述给出结构性视角：[[source-st-foundation-models-survey|STFM 综述]]（A*STAR 2025）提出领域/空间/时间/尺度四维泛化框架，指出现有 STFM 碎片化为交通与天气两大阵营、空间覆盖严重偏向少数大城市[^src-st-foundation-models-survey]；[[source-stfm-pipeline-review|Pipeline 综述]]（2025）把 STFM 生命周期拆为数据协调—模型设计—训练目标—适配四阶段，强调数据质量与领域相关性比规模更重要[^src-stfm-pipeline-review]。

**小结**：跨城泛化的关键分歧是"空间信息何时注入"。FactoST 主张时间先训、空间后注（负迁移论据）；OpenCity/STUNet 主张空间必须显式编码；UrbanFM 用 scaling 压缩空间编码的权重。详见 [[spatio-temporal-foundation-model-landscape|时空基础模型全景]]。

---

## 矛盾五：窗口 vs 记忆——模型能记住多远？

**问题**：固定窗口模型看不到窗口外的全局历史模式。检索增强用"参数外记忆"补上这一块——但记忆什么时候可靠？

[[source-raf|RAF]]（2024）论文自称首个系统化把 RAG 用于时序基础模型的框架：给定序列末尾 motif，在历史中检索最相似 motif 及其后续，拼接后送入冻结 TSFM 零样本预测[^src-raf]。论文的核心发现是检索增强是大型 TSFM 的涌现能力——Chronos Mini 无法执行 TS-R，而 Small/Base 可以；大模型获益更显著，与 LLM 的 RAG 规模定律一致[^src-raf]。

[[rast|RAST]]（AAAI 2026）论文自称首个把 RAG 引入时空预测的通用框架：解耦编码器（2D 卷积时间 + 图变换空间）、上下文感知查询生成器、FAISS 时空检索库、信息论 top-k 检索、通用骨干交叉注意力融合[^src-rast][^src-retrieval-augmented-st-traffic]。论文在信息论上论证它把互信息容量从 H(θ) 扩展到 H(θ)+H(M)，检索复杂度 O(k log M + kd) 显著低于图注意力 O(N²)[^src-rast]。论文报告在 6 个数据集 21 个基线上 SOTA，消融显示查询生成器最关键（移除 MAE 退化 25.6%）[^src-rast][^src-retrieval-augmented-st-traffic]。

[[gtr|GTR]]（ICLR 2026）用可学习全局周期嵌入做即插即用检索：按绝对位置检索周期参考段与局部输入融合，论文报告 PEMS03 上 iTransformer MSE 降 62.2%，且短窗口下仍稳定[^src-gtr]。

检索的失效条件是 [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26|SARAF]]（KDD 2026）的核心发现：相似度检索的可靠性强依赖平稳性——论文报告平稳数据集 Electricity 上输入-未来相似度排名 Spearman ρ=1.000，非平稳 Exchange 上仅 0.285；SARAF 用时间对齐 + 平稳性控制的多样性 MMR 选择 + 自适应聚合，论文报告平均 MSE 较 RAFT 降 3.85%[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

[[source-predicting-the-future-by-retrieving-the-past-aaai2026|PFRP]]（AAAI 2026）用预测对比学习（按未来相似性而非输入相似性聚类）+ K-medoids 构建全局记忆库，与任意局部模型动态融合，论文报告 Traffic/Electricity 上平均提升 17.4%/10.1%[^src-predicting-the-future-by-retrieving-the-past-aaai2026]。

**小结**：检索增强沿着"显式历史库（RAST/RAF）→ 可学习周期嵌入（GTR）→ 蒸馏记忆（PFRP）"演进，并开始处理非平稳下的检索可靠性问题（SARAF）。

---

## 矛盾六：静态 vs 漂移——分布偏移是一等公民吗？

**问题**：交通网络的节点会增删、分布会逐年漂移，IID 假设在真实部署中不成立[^src-stop]。持续学习、OOD 泛化、因果/物理路线各怎么回应？

[[team|TEAM]]（PVLDB 2024）论文自称首次把演化路网预测形式化为图快照序列：混合 Conv+Attention 架构 + Wasserstein 度量选择稳定/不稳定节点分区排练 + EWC 正则，仅训练演化部分，复杂度 O((ΔN+|B|)²)[^src-team]。[[eac|EAC]]（ICLR 2025）冻结骨干 STGNN，仅用可扩展的连续 prompt 参数池适配流式数据，推导出 expand（异构性驱动增长）与 compress（低秩近似，约 41% 参数缩减）两条原则[^src-eac]。[[stbp|STBP]]（ICLR 2026）把 EAC 推向"纯扩展"：固定通用骨干 + 增量扩展的上下文模式库，论文报告 PEMS-Stream 上 MAE 比 EAC 降 21.44%[^src-stbp]。详见 [[continual-spatio-temporal-forecasting|持续时空预测]]。

[[st-ood|ST-OOD]] 基准（IEEE TMC 2025）系统评估跨年自然偏移：六个城市场景训练年 Y、测试年 Y+1 同日历窗，论文报告 OUT 相对 IN 的 RMSE 升幅约 40%–116%；多数复杂模型 OUT 上不如简单 MLP，STID 式轻量架构平均排名靠前，0.2–0.3 dropout 是廉价的 OUT 提升[^src-st-ood]。这印证了 [[stop|STOP]] 的诊断：节点间消息传递机制本身是 OOD 脆弱的来源[^src-stop]。

因果与物理路线试图学不变结构：[[causalx|CausalX]]（ICML 2026）用 Granger 因果/do-calculus/TDMI/VAE 四种约束监督动态因果图 + 扩散图精炼，论文报告即插即用地提升多模态时空预测并产出可解释因果结构[^src-causalx]。[[stpde|STPDE]]（ICML 2026）把时空预测重述为非齐次 PDE 演化：不变扩散算子（Green 函数 = 线性注意力，O(N)）+ 环境基底流形参数化空间异质性 + 随机环境扰动正则，论文报告 OOD 场景下一致优于 GWNet/D2STGNN 等强基线[^src-stpde]。[[mmckm|MMCKM]]（ICLR 2026）用微-宏耦合 Koopman 建模：车辆级图 PDE（反对称平流算子 + 半正定扩散算子）提升到线性观测空间，无需历史轨迹即可演化，意图判别器（MoE）在 5 个有界 Koopman 算子间路由[^src-mmckm]。更广阔的 OOD 应对谱系见 [[ood-generalization|OOD 泛化]]。

**小结**：鲁棒性从"事后修补"变为"架构原生"——集中式交互替代节点间消息（STOP）、冻结骨干 + 可扩展记忆（EAC/STBP）、不变算子 + 环境解耦（STPDE/MMCKM）。ST-OOD 的基准结果同时给了一个警示：多数复杂模型在跨年偏移下不如简单 MLP[^src-st-ood]。

---

## 矛盾七：数值 vs 语义——LLM 的世界知识能用吗？

**问题**：交通预测是纯数值任务，但城市语义（POI、路网、事件）藏在文本里。LLM 怎么接入？四条路线各有代价。

### 路线 A：LLM 直接做时空推理（ST-LLM）

[[urbangpt|UrbanGPT]]（KDD 2024）论文自称首个时空 LLM：多级门控膨胀卷积编码器（刻意无图，零样本场景目标区域空间关系未知）+ 时空-文本对齐投影 + 自然语言 prompt（时间/POI 空间/任务）+ 回归层[^src-urbangpt]。论文报告 LLM 不直接输出数值（移除回归层是最严重退化），而是输出富含推理的隐向量由回归层映射[^src-urbangpt]。论文报告零样本跨区域 NYC-taxi MAE 6.16 vs ASTGCN 9.75，但 7B 参数推理 174s/传感器，难以大规模部署[^src-urbangpt]。

### 路线 B：冻结 LLM 作为语义增强器

[[time-llm|Time-LLM]]（ICLR 2024）用 patch reprogramming（交叉注意力把 patch 对齐到词嵌入空间）+ prompt-as-prefix 把冻结 LLM 复用为通用时序预测器，论文报告仅训练 ~6.6M 参数（0.2%），7/8 数据集 MSE 最优[^src-time-llm]。[[fstllm|FSTLLM]]（ICML 2025）针对 few-shot 时空场景：冻结 LLaMA-2-7B 编码节点文本生成语义邻接矩阵（α-Entmax 稀疏化），再 QLoRA 微调 LLM 用六段 prompt 校准数值预测，论文报告 3 天数据训练击败 30 天数据基线，MAPE 降约 30%[^src-fstllm]。[[source-geolocation-llm-st|LLMGeovec]]（AAAI 2025）用 LLM + OpenStreetMap 生成训练无关的地理定位嵌入，拼接进 DCRNN/STGCN/GWNet 等 STGNN 即提升性能，论文报告简单 MLP + LLMGeovec 可达 GNN 相当水平（最高 +26.53%）[^src-geolocation-llm-st]。

### 路线 C：LLM 作为任务统一器

[[bigcity|BIGCity]] 用 GPT-2 + LoRA + 任务 prompt 统一轨迹与交通状态[^src-bigcity]。[[streasoner|STReasoner]]（2026）走显式时空推理：patchify + MLP 编码时序 token 与文本 token（含图结构文本）交错输入 LLM，S-GRPO 用"有空间 vs 无空间"对比奖励激励空间推理，论文报告 8B 模型在病因/实体/相关性推理任务上超越 GPT-5.2 且成本仅 0.004×[^src-streasoner]。

### 路线 D：LLM 与主线趋势的张力

LLM 路线也暴露出与主线趋势的张力：UrbanGPT 的 174s/传感器推理与效率趋势相悖[^src-urbangpt]；而 [[st-ood|ST-OOD]] 的证据表明 LLM 辅助（如 UrbanGPT 式零样本）是论文列举的未来方向之一[^src-st-ood]。

**小结**：LLM 融合的四条路线差异在"LLM 承担什么角色"——直接推理（UrbanGPT）、语义增强（Time-LLM/FSTLLM/LLMGeovec）、任务统一（BIGCity/STReasoner）。共同代价是推理成本与部署规模。

---

## 两条贯穿性主线

以上七个矛盾不是孤立的，重读会发现两条反复出现的张力：

### 主线一：复杂度的"还债"

二次复杂度（O(T²) 注意力、O(N²) 自适应邻接）在多个时代被同一批手段偿还：SSM 的线性状态空间（[[source-stg-mamba|STG-Mamba]] 论文报告 O(N) 复杂度并一致超越 STAEformer 等基线[^src-stg-mamba]）、核近似的线性化卷积（BigST 的 PRF 分解[^src-bigst]、[[ragc|RAGC]] 的余弦相似度算子[^src-ragc-efficient-traffic-forecasting]）、空间分块（[[patchstg|PatchSTG]] 的 KDTree 分块，论文报告 CA 上 10× 训练加速、4× 内存节省[^src-patchstg]）、代理 token（[[source-fast-long-horizon-forecasting|FaST]] 的 O(Na) 代理 token 注意力[^src-fast-long-horizon-forecasting]、[[stop|STOP]] 的集中式低秩交互[^src-stop]）、时间折叠（[[visifold|VisiFold]] 把 T 个快照折叠成单图，token 数 N×T→N，论文报告比 STAEformer 训练快约 7×、省内存 4×，80% 掩码率仍保持性能[^src-visifold]）。[[bigst|BigST]]（PVLDB 2024）把长序列建模拆为可缓存预计算特征提取器 + 线性化空间卷积，复杂度 O(N)，论文报告扩展到约 10 万节点，比 GWNet 训练加速 2.3–20.6 倍[^src-bigst]。[[graphsparsenet|GSNet]]（PVLDB 2025）观察到训练良好的自适应邻接高度稀疏，用两个小矩阵（C×C 低维邻接 K + 组合系数 U）在压缩空间完成图运算，论文报告 CA 上 MAE 19.76 SOTA 且训练比 BigST 快 3.51 倍[^src-graphsparsenet]。SSM 路线本身是这一主线的代表：[[s-mamba|S-Mamba]]（Neurocomputing 2024）论文自称首个 Mamba MTSF 基线，双向 Mamba 编码变量间相关性（VC）、FFN 编码时间依赖（TD），论文报告消融证明"VC 用 Mamba、TD 用 FFN"是最优分工[^src-s-mamba]。[[dst-mamba|DST-Mamba]]（AAAI 2025）针对长程交通预测的时空纠缠：移动平均分解出趋势（多尺度线性预测）与季节（双向 Mamba 在空间节点维度编码），论文报告 PEMS03/04/07/08 上多数 SOTA，消融显示移除分解 MSE +16.8%、移除季节组件近乎崩溃（0.513）[^src-dst-mamba]。[[rivermamba|RiverMamba]]（NeurIPS 2025）把 Mamba 推到全球尺度：用空间填充曲线将河网串行化为 1D 序列，双向 SSM 在 0.05° 全球网格上做 7 天洪水预报，论文报告 F1 全面超越 GloFAS 物理模型与 LSTM[^src-rivermamba]]。[[gamma-net|GAMMA-Net]]（arXiv 2026）交错堆叠 GAT 与时间轴/空间轴 Mamba（L=3 双阶段闭环），论文报告 6 个基准上最多降 16.25% MAE；移除双轴 Mamba 扫描导致 MAE 飙升 44%–45%[^src-gamma-net]。论文自述仍依赖预定义图结构，且未探讨快速拓扑变化[^src-gamma-net]。

### 主线二：外生信息的"加冕"

事件与多模态从"可选插件"变成"标准输入"。事故被显式建模：[[conformer|ConFormer]]（KDD 2026）用事故感知图传播 + 引导层归一化（GLN 条件仿射参数），论文报告事故场景最高提升 10.7%[^src-conformer]；[[igstgnn|IGSTGNN]]（KDD 2026）用 ICSF 空间融合 + TIID 高斯时间衰减显式建模事件影响的传播与消散，论文报告 Alameda 上平均 MAE 降 5.65%[^src-incident-guided-st-forecasting]。多模态从 MoST 的 SNR 门控选模态[^src-most]到 LLMGeovec 的文本地理嵌入[^src-geolocation-llm-st]，外生信息正在成为标准输入而非可选插件。

---

## 未解决的问题

- **动态图与事件闭环**：GAMMA-Net 与 STUNet 均依赖预定义图，动态图/事故冲击未作为主设定[^src-gamma-net][^src-stunet]；IGSTGNN 仅处理最新时间步事件，历史事件影响假设隐含在近期数据中[^src-incident-guided-st-forecasting]。
- **推理成本**：LLM 路线（UrbanGPT 174s/传感器）与大规模部署矛盾[^src-urbangpt]；PLM backbone 引入额外推理开销[^src-std-plm]。
- **检索的可靠性边界**：非平稳数据上"相似的过去≠相似的未来"（Exchange ρ=0.285）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]；冷启动与全新场景下记忆库失效[^src-rast]。
- **评估标准化**：STFM 空间覆盖严重偏向少数大城市，缺乏统一基准[^src-st-foundation-models-survey]；跨年 OOD 下复杂模型普遍退化[^src-st-ood]。
- **空间信息的最终形态**：如果 MiniST/KD-Tree 隐式编码思路被验证，未来 STFM 可能不再需要邻接矩阵、Laplacian 特征向量与 GCN[^src-urbanfm]。

---

## 引用源

[^src-stgcn]: [[source-stgcn]]
[^src-dcrnn]: [[source-dcrnn]]
[^src-gwnet]: [[source-gwnet]]
[^src-astgcn]: [[source-astgcn]]
[^src-mtgnn]: [[source-mtgnn]]
[^src-patchtst]: [[source-patchtst]]
[^src-itransformer]: [[source-itransformer]]
[^src-zeng-2022-are-transformers-effective]: [[source-zeng-2022-are-transformers-effective]]
[^src-autoformer]: [[source-autoformer]]
[^src-timesnet]: [[source-timesnet]]
[^src-staeformer]: [[source-staeformer]]
[^src-testam]: [[source-testam]]
[^src-stg-mamba]: [[source-stg-mamba]]
[^src-dst-mamba]: [[source-dst-mamba]]
[^src-s-mamba]: [[source-s-mamba]]
[^src-rivermamba]: [[source-rivermamba]]
[^src-gamma-net]: [[source-gamma-net]]
[^src-stunet]: [[source-stunet]]
[^src-most]: [[source-most]]
[^src-opencity]: [[source-opencity]]
[^src-unist]: [[source-unist]]
[^src-urbandit]: [[source-urbandit]]
[^src-factost]: [[source-factost]]
[^src-urbanfm]: [[source-urbanfm]]
[^src-stfm-pipeline-review]: [[source-stfm-pipeline-review]]
[^src-st-foundation-models-survey]: [[source-st-foundation-models-survey]]
[^src-std-plm]: [[source-std-plm]]
[^src-urbanpg]: [[source-urbanpg]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-diffstg]: [[source-diffstg]]
[^src-ustd]: [[source-ustd]]
[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
[^src-craft]: [[source-craft]]
[^src-tedm]: [[source-tedm]]
[^src-flowts]: [[source-flowts]]
[^src-tsflow]: [[source-tsflow]]
[^src-sundial]: [[source-sundial]]
[^src-loft]: [[source-loft]]
[^src-rast]: [[source-rast]]
[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
[^src-retrieval-augmented-st-traffic]: [[source-retrieval-augmented-st-traffic]]
[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
[^src-gtr]: [[source-gtr]]
[^src-patchstg]: [[source-patchstg]]
[^src-bigst]: [[source-bigst]]
[^src-graphsparsenet]: [[source-graphsparsenet]]
[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-visifold]: [[source-visifold]]
[^src-team]: [[source-team]]
[^src-eac]: [[source-eac]]
[^src-stbp]: [[source-stbp]]
[^src-stop]: [[source-stop]]
[^src-st-ood]: [[source-st-ood]]
[^src-causalx]: [[source-causalx]]
[^src-stpde]: [[source-stpde]]
[^src-mmckm]: [[source-mmckm]]
[^src-urbangpt]: [[source-urbangpt]]
[^src-time-llm]: [[source-time-llm]]
[^src-bigcity]: [[source-bigcity]]
[^src-fstllm]: [[source-fstllm]]
[^src-geolocation-llm-st]: [[source-geolocation-llm-st]]
[^src-streasoner]: [[source-streasoner]]
[^src-conformer]: [[source-conformer]]
[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-raf]: [[source-raf]]
[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
