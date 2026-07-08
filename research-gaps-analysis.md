# 研究空白与实验完整性分析

> 分析范围：10 篇核心论文 + 3 篇综述论文
> 分析日期：2026-07-08
> 方法：通过 pdftotext 提取全文，聚焦实验部分的宣称-实际差距分析

---

## 一、各论文实验审视

---

### 1. ExoST (2509.05779) — "Plug-and-Play" 外生框架 ⚠️ arXiv only, 未经同行评审

**宣称：** "plug-and-play" 外生集成框架，在任何 ST backbone 上实现"consistent improvements"，"most exceed 20%"（§5.2）。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **提升幅度** | ">20%" 是 MRE（相对误差率）的提升，非 MAE/MSE 绝对提升。从 Table 2 看，在 Speed-19 上 ExoST 的 MAE=14.52（TiDE=14.73，NBEATSx=13.88——ExoST 甚至略差于 NBEATSx），在 AQI-19 上与 TimeXer 对比（9.33 vs 9.65）仅有 ~3% 绝对改善。大百分比来自 MRE（相对误差率），这是一个经过归一化的容易放大的指标。 |
| **数据集多样性** | 仅 4 个任务/2 个场景（马德里空气质量 + 交通速度/强度），实际上都来自同一个 Madrid 系列数据集（Zenodo 7308425）。与论文声称的 "real-world spatio-temporal datasets" 相比，缺失跨城市、跨气候、跨领域验证。 |
| **"长期"定义** | 24-step horizon (1 天)，多步滚动到 72-step (3 天)。这在 traffic/weather 领域属于短期到中期预测。实践中长期预测通常以周/月为单位。 |
| **Baseline 选择** | 比较了 TiDE、NBEATSx、TimeXer、ChronosX、MAGCRN。缺失重要的 ST foundation models 如 UniST、OpenCity、UrbanGPT，以及多模态方法如 ExoLLM。ChronosX 本身是 foundation model 的外生适配版，但论文承认 ChronosX 在多个任务上 "frequently suffers from non-convergence"——这是否是一台公平的比较？ |
| **消融设计** | 从数据视角（过去/未来/日期外生变量）和模型视角（Selector/Balancer）分别做消融，设计清晰。但数据消融中"日期外生变量单独效果差"——这可能是 trivial 的发现。模型消融没有 baseline 对比，只有 own variants。 |
| **效率分析** | 宣称 lightweight，但只报告了参数数，缺少与 baseline 的实际推理时间对比。 |
| **代码可用性** | 未在论文中提及代码库。 |

**评分卡：** ✔ 消融设计合理 | ✘ 数据集狭窄（单城市系列） | ✘ 宣称成绩夸大（MRE 而非 MAE/MSE） | ✘ 缺少跨领域验证 | ✘ 无代码 | ✘ arXiv only (未经同行评审)

---

### 2. E²-CSTP (2505.17637) — 因果多模态预测

**宣称：** "up to 9.66% improvement in accuracy" + "17.37%–56.11% speedup" + 因果推理。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **9.66% 提升** | 从 Table 1 看，在 Terra 上 E²-CSTP (MAE=2.43) 对比最佳 baseline D²STGNN (MAE=2.52) = 3.57% 提升；在 BjTT 上 (3.56 vs 3.62) = 1.66% 提升。9.66% 是最佳场景（可能是 GreenEarthNet/BikeNYC 上的某一指标），但不是一致水平。 |
| **因果有效性** | DeepSHAP 用于识别因果区域——这是一个 post-hoc 解释方法，**不是学习的因果图**。因果干预模块的消融（w/o CI）确实导致性能下降（从 Fig.3 看 Terra 上 MAE 从 ~2.43 升到 ~2.55，约 5%），但无法区分这归因于因果推理还是额外的正则化效应。 |
| **速度提升** | 17.37%–56.11% 的速度对比是**针对 Transformer 预测模块变体**（w/ In, w/ Auto, w/ FED, w/ iTrans），不是完整模型对比。完整的 9 baseline 效率对比缺乏数值表，只有 Fig.4 的柱状图。在 BikeNYC 上由于是单模态，E²-CSTP 更快，但这与模型设计无关。 |
| **数据集** | 4 个数据集（Terra, BjTT, GreenEarthNet, BikeNYC）来自不同的领域，多样性好。但 BikeNYC 是单模态（无文本/图像），因此不测试多模态能力。GreenEarthNet 没有 T3 和 FNF baseline（缺少文本模态），造成比较不完整。 |
| **基线排除** | T3 和 FNF 在 GreenEarthNet 和 BikeNYC 上被排除，使得在这些数据集上的对比不公平。另外缺失 LLM-based 方法如 TimeLLM 的比较。 |
| **消融** | 6 组件消融（Text, Image, DeepSHAP, CI, GCN, Mamba），框架完整。但所有消融在同一数据集上用 bar chart 展示，缺乏数值精确性。且没有讨论组件之间的交互效应。 |
| **超参数敏感性** | λ 和 β 的搜索只在 {0, 0.25, 0.5, 0.75, 1} 五个值上，且不同数据集取值不同——这表示模型需要 per-dataset tuning。 |
| **可重复性** | 提到 "implementation details" 在附录但没有明确声称代码发布。 |

**评分卡：** ✔ 数据集多样性好 | ✔ 消融设计较完整 | ✘ 9.66% 提升是 cherry-pick | ✘ 因果识别靠 post-hoc 而非 learnable causal graph | ✘ 效率对比有误导性（只对预测模块而非完整 pipeline） | ✘ 缺乏与 LLM-based 方法的比较 |

---

### 3. ExoLLM (exollm-www-2025.pdf) — LLM + 外生文本

**宣称：** "31.2% MSE and 19.8% MAE improvement over TimeLLM" + "9.1% and 4.1% over TimeXer"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **显著改善 vs TimeLLM** | TimeLLM 是一个通用的 LLM-based 时间序列模型，**并非为外生变量设计**。以 TimeLLM 作为 baseline 来比较外生变量处理能力是不公平的——TimeLLM 甚至没有设计来处理外生输入。这个 31.2% 的改善数字有噪音。 |
| **vs TimeXer** | 9.1%/4.1% 的改善更有说服力（TimeXer 也是外生感知的）。从 Table 3（long-term）看，在 ETTh1 上 ExoLLM (MSE=0.084) vs TimeXer (MSE=0.094) 确实有改善，但在 ETTm1 上 (0.057 vs 0.062) 和 ETTm2 上 (0.144 vs 0.156) 差异更小。在 Weather 和 Traffic 等更大的数据集上差距可能进一步缩小。 |
| **Short-term 宣称** | 对比 SCINet（非外生感知模型）报告的 35.5%/46.1% 改善有类似问题——SCINet 不是外生感知的。 |
| **LLM vs 简单编码** | 消融（§5.4）测试了 MGP/MTI/DT2 A/TPT 等组件，但没有回答核心问题：**LLM 的能力是否被充分使用？** 即，将 LLM 替换为一个更小的文本编码器（如 BERT-Small 或简单的 TF-IDF 嵌入 + MLP）会如何？没有这个实验，我们无法知道 LLM 的额外计算成本是否值得。 |
| **数据集多样性** | 12 个数据集覆盖了常见的 ETT 系列、Weather、Traffic、ECL 等，以及 5 个 EPF（电力价格）短期数据集。多样性可接受，但这些都是基准测试数据集，不是真正的外生变量场景。外生变量是从这些数据集中**检索的文本描述**（"collected from web"），这种外生变量的构建在现实场景中可能不可复制。 |
| **零样本测试** | Table 6 显示 ExoLLM 在 ETT family 内跨数据集迁移（ETTh1→ETTh2, ETTm1→ETTm2 等）。但这些是同一数据生成过程的不同版本，不是真正的跨域零样本。 |
| **可重复性** | 未提及代码发布。 |

**评分卡：** ✔ 12 数据集覆盖度好 | ✔ 消融设计清晰 | ✘ LLM vs 更简单编码基线缺失（关键消融） | ✘ 对 TimeLLM/SCINet 的对比不公平 | ✘ 外生变量是人工构建的文本 | ✘ 零样本局限在 ETT 家族内 |

---

### 4. ST-Vision-LLM (2510.11282) — 视觉 LLM 时空预测

**宣称：** "15.6% in long-term prediction accuracy" + "30% average improvement in cross-domain few-shot"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **15.6% 宣称** | 基准测试仅在一个数据集（Milan-Internet）上进行。单数据集上的 15.6% 提升缺乏跨数据集的统计效力。从 Table I 看，在 Milan-Internet K=60（论文定义的"long-term"=10小时）上，ST-Vision-LLM 的 NRMSE=0.3664 vs 次优模型 GWNET=0.5579——但 GWNET 在这个 horzion 上的表现确实异常差。在所有 baseline 上，许多方法的 NRMSE 超过 0.5（接近随机）。这表明问题可能设置不当或数据预处理导致部分基线表现不佳。 |
| **"long-term"定义** | K=60 steps × 10 min = 10 小时。这**不是**时空预测领域通常意义上的 long-term。标准定义中，long-term 通常是 days/weeks。论文标题带"long-term"但实际预测范围不到半天。 |
| **数据集规模** | 仅 2 个月数据（2013-11-01 到 2014-01-01），48 天训练 + 7 天验证 + 7 天测试。极短的时间跨度不能充分评估模型在不同季节/日期间的变化。数据通过线性插值填补缺失值（论文承认有 "discontinuous traffic data"），且评估时排除插值位置——但这意味着**不同模型在数据子集上的比较**，公平性存疑。 |
| **Baseline 选择** | 包含 ARIMA、ST-ResNet（2018）等旧方法，但缺失许多近年重要的 ST baseline：STAEformer、D2STGNN、STG-Mamba 等。LLM baselines 包括 GCNGPT、GATGPT——这些不是主流比较对象。缺失 TimeXer、PatchTST 等经论文验证的模型。 |
| **Cross-domain 定义** | "Cross-domain" = Milan vs Trentino，两者都是同一移动通信数据集中的地理区域。不是真正的跨域（如交通→天气→能源）。因此"30% cross-domain improvement"被大幅夸大了。 |
| **Few-shot 实验** | 5%/10% 训练数据消减。但原数据集仅 48 天训练数据，5% = ~2.4 天——在如此短的数据上做 few-shot 评估的可靠性存疑。 |
| **代码** | 未提及。 |

**评分卡：** ✘ 单数据集主实验 | ✘ "long-term"只定义了 10 小时 | ✘ baseline 过时 | ✘ cross-domain = 同一数据集不同区域 | ✘ 数据量极端有限（2 个月） | ✘ 无代码 |

---

### 5. Aurora (2509.22295) — 多模态时间序列基础模型

**宣称：** "first Multimodal Time Series Foundation Model" + "zero-shot inference" + "SOTA on 5 benchmarks"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **"First" 宣称** | 论文称 Aurora 是 "first multimodal time series foundation model"，但同期/之前已有工作（如 CALF、GPT4MTS 等）也在多模态时间序列上做预训练。这个宣称取决于对"foundation model"的严格定义（大规模预训练+零样本），但并非毫无争议。 |
| **零样本验证** | 在 TimeMMD 上与 unimodal FMs 的直接比较（Table 1）：Aurora zero-shot 对比 Sundial 改善 27.0% MSE，对比 VisionTS 改善 31.2%。但需注意这些 unimodal FMs 不使用文本/图像信息——比较本身不对称。Aurora 的优势可能来自额外模态而非更好的时间序列建模。更公平的消融（Aurora without multimodal inputs vs 其他 FMs）没有充分展示（Table 2 是 unimodal evaluation 但只有 2 个 benchmark）。 |
| **Few-shot vs Full-shot** | 10% 数据微调的 Aurora 对比 full-shot 的 GPT4MTS/CALF 更好（MSE 降低 12.8%/24.5%）。但这个比较本身有价值但也存在问题：full-shot supervised 模型是为了完全监督而设计的，而 Aurora 利用了预训练。10% 的优势展示了预训练的价值，但不一定是多模态带来的。 |
| **Domain 多样性** | 5 个 benchmarks（TimeMMD, TSFM-Bench, ProbTS, TFB, EPF）覆盖多种领域，但 TimeMMD 中的数据集本身有限。宣称 "cross-domain" 但许多数据集相似度高。 |
| **消融局限** | 消融（§4.6）只对 Modality-Guided Multi-head Self-Attention 做了一种变体检查。缺乏对预训练数据量、模型规模、模态组合策略的系统消融。 |
| **预训练语料库** | Cross-Domain Multimodal Time Series Corpus 的构成不够透明。"collected from open-source datasets" + "LLM-generated textual descriptions"——自动生成的文本质量未知，可能存在幻觉。作者宣称评估数据集被严格排除，但潜在的**数据污染风险**依赖于自我声明。 |
| **计算开销** | 作为 FM，训练和推理成本未详细报告。使用 "10% of data" 与 full-shot 比较是一个优势，但 fine-tuning 本身的开销也未提及。 |
| **代码** | 未提及。 |

**评分卡：** ✔ 5 benchmarks 评估较广泛 | ✔ 零样本+少样本设置合理 | ✘ "first" 宣称有争议 | ✘ 消融不够深入 | ✘ 预训练数据透明度低 | ✘ 无代码 | ✘ 不对称比较（额外模态 vs unimodal FMs） |

---

### 6. CaST (2309.13378) — 因果 OOD 泛化

**宣称：** "causal treatments" for OOD generalization + "consistently outperforms existing methods"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **OOD 验证** | 仔细检查 §5，OOD/泛化实验就是**标准的时序训练/测试分割**（24→24 步）。没有可控的 OOD 场景，比如在周末数据上训练、在假期数据上测试，或在城市 A 训练在城市 B 测试。论文的"OOD generalization"实际上等同于标准泛化性能。 |
| **改善幅度** | 从 Table 1 看：PEMS08 上 CaST (MAE=16.44) vs AGCRN (17.06) = 3.6% 改善。AIR-BJ (22.90 vs 23.43) = 2.3%。AIR-GZ (11.66 vs 12.43) = 6.2%。这些是扎实但不突出的改进，**不足以支撑 "causal treatment" 这一重大宣称**。 |
| **因果图质量** | 论文的 SCM 是**假设的**（Figure 2a），不是从数据中学习的。变量 E（环境）和 C（空间因果）被假设为独立——这个假设在现实 ST 场景中几乎肯定不成立（例如天气同时影响交通流量和空气质量）。back-door 和 front-door adjustment 严重依赖于这些假设。论文没有提供验证这些假设的实验。 |
| **消融** | 消融（Figure 5a）检查了 Env/Ent/Edge 三个组件，显示各有贡献。但是 "w/o Env" 的退化较小——挑战了 OOD 声明：如果环境变量是核心贡献，移除它应该导致更大退化。 |
| **"Dynamic causal relations"** | 声称建模因果关系，但实质是使用 edge-level convolution 的权重作为因果强度。**注意力不等于因果**（"no causation without manipulation"），这点论文没有充分讨论。 |
| **数据集** | 3 个标准数据集（PEMS08, AIR-BJ, AIR-GZ），都是 24→24 的设置。没有扩展到更长 horizon 或多个领域。 |
| **代码** | 未提及。 |

**评分卡：** ✘ "OOD" 被定义为标准时序泛化 | ✘ 因果图是假设的不可验证 | ✘ 改善幅度小 | ✘ 消融不支持 OOD 宣称 | ✘ 因果 claim 需要更强证据 | ✘ 无代码 |

---

### 7. ClimaX (2301.10343) — 天气/气候基础模型

**宣称：** "superior performance on benchmarks" + "generality from heterogeneous datasets"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **宣称-实际差距** | 论文声称 SOTA on ClimateBench 和 competitive on weather forecasting。实际阅读 §4 发现：在 global forecasting 上 ClimaX 在部分变量/horizons 优于 baselines，但不是一致的。论文诚实地报告了 "competitive" 而非 "dominating"。在气候 projection 和 downscaling 上，评估是定性的或有限指标。 |
| **预训练策略** | 使用 CMIP6 仿真数据做预训练是新颖的，但这也意味着模型学习的是**模拟物理**而非真实天气模式。模型在 ERA5 真实数据上 fine-tune，但预训练-微调之间的 gap（模拟 vs 真实）可能限制 transfer 效果。论文没有提供消融来量化 CMIP6 预训练相对于仅 ERA5 训练的增益。 |
| **任务多样性** | 覆盖多种任务（global/regional forecasting, climate projection, downscaling, sub-seasonal to seasonal），确实展示了 generality。但每个任务上的实验设置不同，部分任务的评估不完整（如 downscaling 只有定性结果）。 |
| **消融设计** | §4.6 的消融（separate vs joint finetuning, iterative vs direct forecast, per-lead-time vs all）实践价值高，但不是典型的方法消融。缺少架构消融（如不同 patch size、不同 embedding 策略的影响）。 |
| **代码可用性** | 论文声明代码可用。这是本集合中少数提供代码的论文之一。 |
| **缩放定律** | §4.5 的缩放实验使用了明确的 scaling law 分析（数据集数、模型大小、分辨率），这是值得肯定的。但缩放实验只在 weather forecasting 上进行，未验证在 climate tasks 上的 scaling。 |

**评分卡：** ✔ 任务覆盖广 | ✔ 代码可用 | ✔ 缩放分析 | ✔ 诚实地报告结果 | ✘ 预训练增益未通过消融量化 | ✘ CMIP6→ERA5 的 gap 存在风险 | ✘ 部分任务评估不完整 |

---

### 8. STG-Mamba (2403.12418) — Mamba 时空图学习

**宣称：** "first exploration of SSSM for STG learning" + "linear complexity" + "consistently outperforms"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **线性复杂度声称** | SSSM 组件（ST-S3M）确实有线性复杂度，但 STG-Mamba 还包括 KFGN（Kalman Filtering GNN），其中 GNN 组件在图大小上仍具有二次或 O(E) 复杂度。**总体复杂度不是线性的**——论文的线性宣称仅适用于选择性 SSM 部分而非整个模型。对比 Transformer-based 方法的效率优势来自于 SSM 的 GPU 优化扫描，但加上 GNN 后，这个优势需要更仔细的核算。 |
| **效率分析** | Appendix 提供了理论计算复杂度，但实际推理时间只在特定设置下报告。缺少与 baseline 在大图上的实际训练/推理吞吐量对比。 |
| **改善幅度** | 从 Table 的数值看，STG-Mamba 在 PeMS04、HZMetro、KnowAir 上确实实现了 SOTA 或接近 SOTA 的性能，但改善幅度通常是 1-3% 的 MAE/RMSE，远非革命性。在 PeMS04 (Flow) 的 MAPE 上甚至不如部分 baseline。 |
| **消融** | 5 个变体的消融（§5.7）涵盖了 KF-Upgrading, Dynamic Filter, GSSSM, ST-S3M 的逐步消融。结果理想：每个组件都重要。但需要注意的是，去除 ST-S3M 后模型退化到 "plain GNN"——但 plain GNN 训练配置可能与优化后的 STG-Mamba 不同，这不是等量比较（不公平的优势）。 |
| **数据集** | 3 个 STG 数据集（交通/地铁/天气），多样性一般。缺乏跨域验证（如能源、社交网络）。 |
| **卡尔曼滤波声称** | KFGN 使用卡尔曼滤波启发的方法来融合多粒度特征。但该方法是否严格遵循卡尔曼滤波的数学框架（线性高斯状态空间模型）？论文承认是 "optimized version"——实际上可能只是带有可学习权重的 feature fusion，深度学习的通用技巧而非严格的状态估计。 |
| **代码** | 代码在 GitHub 可用（https://github.com/LincanLi98/STG-Mamba）。 |

**评分卡：** ✔ 代码可用 | ✔ 消融设计好 | ✘ "线性复杂度" 过度宣称（仅限 SSM 组件） | ✘ 改善幅度小 | ✘ KF 声称弱化了（就是加权融合） | ✘ 数据集多样性不够 |

---

### 9. TimeXer (2402.19072) — 基于交叉注意力的外生注入

**宣称：** "empowers Transformer with exogenous variables" + "consistent SOTA on twelve datasets"。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **实际改进幅度** | 在 long-term multivariate 基准上（Table 3），TimeXer vs iTransformer 的改进非常温和：ECL (0.171 vs 0.178, +4%), Weather (0.241 vs 0.258, +7%), ETTh1 (0.437 vs 0.454, +4%), Traffic (0.466 vs 0.428——此处 iTransformer 更好)。"Consistent" 但不是大幅的。论文诚实地报告了这些数据。 |
| **外生变量在长期实验中的作用** | Table 3 的实验设置是**多变量预测**而非典型的外生变量场景（所有变量都需预测）。论文在 §5.1 解释这是 "long-term forecasting with exogenous variables on multivariate benchmarks"，但实践中所有变量同时输入，没有区分内生/外生。这不是真正的外生变量场景。 |
| **Short-term EPF 结果** | 在电力价格预测（真正的 FEV 场景）上改进更显著（Table 2, AVG MSE 0.307 vs 次优 iTransformer 0.330）。这才是论文方法的核心场景。 |
| **消融** | 比较了不同的 token 设计（patch/global/variate tokens）和 fusion 方法（add vs concat vs cross-attention）。消融很有说服力，展示了 cross-attention 设计带来的改进。但没有测试核心问题：外生变量到底提供了多少增益？即 "只使用内生信息 vs 完整模型" 的对比。 |
| **全局 token 的有效性** | 引人入胜的设计（learnable global token bridging endogenous and exogenous）。但这个 token 的效果没有在消融中独立测试（w/o global token 的变体）。 |
| **长期预测的虚假外生** | 在 multivariate 基准上的外生变量设置存在问题：如果所有变量都作为 exogenous 输入，但传统上它们被视为等价的变体，TimeXer 的优势可能只是来自更好的多变量建模而非真正的外生处理。 |
| **代码** | 论文和许多同一团队的先前工作一样，未提及代码发布。 |

**评分卡：** ✔ 消融设计有说服力 | ✔ 短期 FEV 结果扎实 | ✔ 诚实地报告改进 | ✘ 长期实验在 multivariate 设置下不算是真正的 FEV | ✘ 缺少"仅内生 vs 完整模型" 对比 | ✘ 缺少全局 token 独立消融 | ✘ 无代码 |

---

### 10. DiffSTG (2301.13629) — 概率时空扩散模型

**宣称：** "first diffusion model for probabilistic STG forecasting" + CRPS 改善 4%-14%。

**实验实际：**

| 维度 | 发现 |
|------|------|
| **概率评估** | CRPS 改善：PEMS08 (-5.6%), AIR-BJ (-4.3%), AIR-GZ (-14.3%)。14.3% 在 AIR-GZ 上不错，但 4-5% 在另两个数据集上较温和。**但没有不确定性校准评估**（可靠性图、置信区间覆盖率等）。CRPS 是单一指标，无法捕捉校准质量。 |
| **采样数量** | 仅 8 个 MC 样本（S=8）。对于扩散模型，8 个样本通常不足以精确估计预测分布。实践中，DDPM-based 方法通常需要 50-1000 步去噪和 50-100 样本。 |
| **Baseline 选择** | 比较了 Latent ODE、DeepAR、TimeGrad、CSDI、MC Dropout。但缺乏现代概率方法如 score-based diffusion、flow matching 或最新的 probabilistic Transformers。此外，STG-specific 概率方法仅 MC Dropout 一个（它们不是为概率建模设计的）。 |
| **数据集** | 3 个数据集（PEMS08, AIR-BJ, AIR-GZ），都是 12→12 设定。缺乏更大规模的验证。 |
| **扩散steps** | 搜索空间小：N ∈ [50, 100, 200]。当前扩散模型通常用 100-1000 步。TimeGrad 的搜索更受限（[50, 100]），这给 TimeGrad 带来不公平劣势。 |
| **计算成本** | 论文承认 TimeGrad 是 "extremely time-consuming" 且做了受限搜索，但 DiffSTG 自身的推理计算成本（多步扩散 + 8 次采样）未与全监督 deterministic 方法对比。 |
| **消融** | 在 AIR-GZ 上做消融（w/o spatial, w/o temporal 等），显示空间和时间组件都有贡献。但与 deterministic STGNN 的比较只做了 STGCN（§5.3），没有与更强的 deterministic 方法如 AGCRN、D2STGNN 对比。 |
| **代码** | 代码在 GitHub 可用（https://github.com/wenhaomin/DiffSTG）。 |

**评分卡：** ✔ 代码可用 | ✔ 首个 STG 扩散模型（novelty 认可） | ✘ 不确定性校准未评估 | ✘ 采样数太少（S=8） | ✘ Baseline 缺失现代概率方法 | ✘ 计算成本与 deterministic 方法的对比缺失 | ✘ 缺乏更大规模验证 |

---

### 综述论文提炼的空白

**1. 2501.09045 — Spatio-Temporal Foundation Models: Vision, Challenges, and Opportunities**

明确指出的空白：
- **当前 STFM 过于狭窄**：没有模型能处理<全范围>的 ST 应用（§IV, p.12）。
- **空间偏见**：现有 ST 数据集过度集中在少数大城市（北京、纽约、伦敦），STFMs 有偏向特定区域的风险（§III-B2, p.10）。
- **跨域泛化不足**：ST 数据高度应用依赖，交通模式与疾病爆发之间的共享模式不明确，负迁移风险高（§III-B1）。
- **未实现真正的基础模型 vision**：截至写作时，没有论文能同时证明 domain generalization、spatial generalization、temporal generalization 和 scale generalization（§IV, p.12）。

**2. 2503.13709 — Multi-modal Time Series Analysis: A Tutorial and Survey**

明确指出的空白：
- **跨模态交互不足**：现有方法在融合多模态信息时仍面临 modality gap、misalignment 和 inherent noise（p.2）。
- **领域泛化是关键挑战**：多模态时间序列中的分布偏移是多层面的，不仅来自时间序列本身也来自其他模态（§7.3, p.22）。
- **缺失和噪声模态的鲁棒性不足**：现有方法处理不完整数据能力有限（§7.4, p.22）。
- **缺乏推理能力**：多模态时间序列与外部知识库和 RAG 系统的结合仍处于早期阶段（§7.1, p.22）。

**3. 2506.01364 — Foundation Models for Spatio-Temporal Data: A Survey and Outlook**

明确指出的空白：
- **数据与模型的弱联系**：原始 ST 数据与预训练 foundation model 之间存在 gap（p.5）。
- **缺少属性考量**：先前的 ST 基础模型未充分考虑 ST 数据的固有特性（空间异质性、时间动态性等）（p.5）。
- **提示工程桥接 Gap**：如何有效地将 ST 数据格式化为 foundation models 可理解的输入是关键挑战（§7.1）。
- **工业需求 vs 学术研究差距**：现有研究未能充分弥合与工业应用之间的差距（p.28）。

---

## 二、共性研究空白

### 空白 1：外生变量的定义模糊且无法泛化

10 篇论文中有 8 篇以不同方式定义"外生变量"：
- ExoST 视其他传感器数据为外生
- TimeXer/ExoLLM 用文本描述作外生
- E²-CSTP 用图像+文本
- ST-Vision-LLM 用网格图像
- Aurora 内置多模态编码

**问题**：没有统一的外生变量处理框架。每个方法为其特定的外生定义精致地设计，但在其他定义上可能完全失效。真正的 "plug-and-play" 需要处理任意模态和任意数量的外生变量。

### 空白 2：长期预测的 horizon 定义不统一

| 论文 | "长期"定义 | 是否真"长期"？ |
|------|-----------|--------------|
| ExoST | 1-3 天 | 短期到中期 |
| E²-CSTP | 12 步 | 短期 |
| ExoLLM | 96-720 步（~3 天-1 月）| 中期 |
| ST-Vision-LLM | 10 小时 | **否，短期** |
| CaST | 24 步 | 短期 |
| TimeXer | 96-720 步 | 中期 |
| DiffSTG | 12 步 | 短期 |
| ClimaX | 多尺度（小时到年）| **是** |
| Aurora | 默认 96-720 步 | 中期 |
| STG-Mamba | 12 步 | 短期 |

大多数自称 "long-term" 的论文实际上只做了中期甚至短期预测。缺乏在周/月/年尺度上的评估。

### 空白 3：因果声称缺乏严格的因果验证

CaST、E²-CSTP 都使用因果语言（"causal treatment"、"causal intervention"）但：
- 没有使用实际的因果发现方法（PC、FCI、LiNGAM 等）
- 因果图是先验假设的，不是从数据中学习的
- 没有进行反事实验证或干预实验
- "因果"效应实际上等同于注意力权重或相关性度量

这是**因果推理领域中公认的错误**：没有不通过操纵（intervention）就能建立因果关系的捷径。

### 空白 4：零样本/泛化宣称与实验不匹配

- CaST：声称 OOD 泛化但只做了标准时序分割
- ExoLLM：声称零样本但只在 ETT 家族内测试
- ST-Vision-LLM：声称 cross-domain 但只是同一数据集的地理区域
- Aurora：声称零样本但未做跨模态域的实际测试

对"泛化"的评估标准严重不足。

### 空白 5：代码和可重复性危机

| 论文 | 代码可用 | 备注 |
|------|---------|------|
| ExoST | ✘ | — |
| E²-CSTP | ✘ | — |
| ExoLLM | ✘ | — |
| ST-Vision-LLM | ✘ | — |
| Aurora | ✘ | — |
| CaST | ✘ | — |
| ClimaX | ✔ | |
| STG-Mamba | ✔ | GitHub |
| TimeXer | ✘ | — |
| DiffSTG | ✔ | GitHub |

**只有 3/10 的论文发布代码**，严重阻碍了结果的验证和比较。

### 空白 6：数据集多样性不足

- 大多数论文在同一组基准（PEMS、ETT、Weather、Traffic、ECL）上评估
- 缺少跨**真实跨域**的评估（如交通→天气→能源→健康）
- 外生变量数据集的发布严重不足（Terra 和 BjTT 是少数例外，但出自同一团队）

### 空白 7：消融实验的设计偏见

常见的消融陷阱：
- 将组件移除比较，但移除后模型没有重新优化架构（不公平比较）
- 缺乏"简化基线"——即用更简单的技术替代复杂组件
- 消融仅在 1-2 个数据集上做（可能 cherry-pick）
- 没有交互效应分析

---

## 三、宣称-实际差距汇总

| 论文 | 宣称 | 实际证据 | 差距严重性 |
|------|------|---------|-----------|
| **ExoST** | "consistent improvements >20%" | >20% 是 MRE（相对指标），MAE 改善通常 2-5%；仅 1 个城市的数据 | **严重** |
| **E²-CSTP** | "up to 9.66% improvement" | 最佳场景 9.66%，多数场景 1.6-3.6%；因果靠 post-hoc DeepSHAP | **中等** |
| **ExoLLM** | "31.2% over TimeLLM, 9.1% over TimeXer" | TimeLLM 不是 FEV 模型，不公平比较；vs TimeXer 改进 4-9%；LLM vs 简单编码未测试 | **严重** |
| **ST-Vision-LLM** | "15.6% long-term" + "30% cross-domain" | 单数据集；"long-term"=10 小时；"cross-domain"=同数据不同区域 | **非常严重** |
| **Aurora** | "first multimodal TS FM" | "first" 有争议；零样本与 unimodal FMs 比较不对称；消融浅 | **中等** |
| **CaST** | "OOD generalization via causal treatment" | OOD = 标准时序分割；因果图是假设的；改善幅度 2-6% | **严重** |
| **ClimaX** | "superior performance" | 多任务但有选择性地好；部分评估定性；诚实地报告了 | **轻度** |
| **STG-Mamba** | "linear complexity" | 仅 SSM 部分线性，GNN 部分不是；KF 声称弱化 | **中等** |
| **TimeXer** | "consistent SOTA on 12 datasets" | 长期改进温和（2-7%）；长期实验不是真正 FEV 场景 | **轻度-中等** |
| **DiffSTG** | "probabilistic ST forecasting" | CRPS 改善 4-14% 但无校准评估；仅 8 样本 | **中等** |

---

## 四、值得追问的问题

以下是针对进一步调查的具体可操作问题：

### 对 ExoST 作者
1. 能否在**不同城市/领域**（如 Terra、BjTT）测试 ExoST 的 universality claim，而不只是马德里数据？
2. 能否提供一个 **ExoST vs 简单基线**（如直接融合外生变量到 backbone）的公平对比，而非仅与基础 backbone 比？
3. 为何 ChronosX 在多个任务上"non-convergence"？这是方法问题还是实现的 bug？

### 对 E²-CSTP 作者
1. DeepSHAP 识别出的"因果区域"是否与**领域知识/专家标注**一致？是否做过任何验证？
2. 如果移除因果干预模块（w/o CI）但保持额外参数，退化幅度是否类似？如何区分因果 vs 参数量的影响？
3. 效率加速的 17.37%-56.11% 是否包括**图像/文本编码器的开销**？

### 对 ExoLLM 作者
1. 如将 LLM 替换为更小的文本编码器（如 DistilBERT 或 simple TF-IDF+MLP），性能差距有多大？LLM 的额外成本是否合理？
2. Short-term 实验中对比 SCINet（非 FEV 模型）是否公平？能否只与其他 FEV 模型对比？
3. 检索到的外生文本能否公开？其他人能否复现同一个外生变量集？

### 对 ST-Vision-LLM 作者
1. 能否提供在**至少 3 个不同领域数据集**上的跨域评估，而非 Milan/Trentino 的局部变体？
2. "10 小时" 作为 long-term 的依据是什么？能否在周/月尺度上测试？
3. 许多 baseline 在该数据上 NRMSE > 0.5——数据是否存在预处理问题？能否提供原始数据质量报告？

### 对 Aurora 作者
1. Aurora 在 TimeMMD 上的零样本优势来自多模态信息还是更好的 time series 建模？能否在**去掉图像/文本输入**后与其他 FMs 对比？
2. 预训练语料库中文本描述的具体生成方法？有无人工抽样验证质量？
3. 能否提供在不同数据集上的**逐条**结果（而非平均）来展示一致性和方差？

### 对 CaST 作者
1. "OOD generalization" 性能能否在**人为构建的 distribution shift** 上验证（如训练/测试在不同季节、不同城市、不同传感器子集）？
2. SCM 中的独立性假设（E⟂C）如何验证？这个假设在您的数据集中成立吗？
3. "因果强度"（causal strength）能否通过**干预实验**验证？如屏蔽某一因果边后观察预测变化？

### 对 STG-Mamba 作者
1. 整个模型的**实际复杂度**（而非仅 SSM 组件）在大图上的表现？与 Transformer-based 方法的真实训练/推理成本对比？
2. Kalman Filter Upgrading 相比简单加权平均（learnable α×x₁ + (1-α)×x₂）的优势？有消融吗？
3. 3 个数据集的改善是否能在**更多不同领域**（能源、社交网络、生物）中复现？

### 对 TimeXer 作者
1. 在长期 multivariate 实验中，外生变量的角色是什么（所有变量互作输入和预测目标）？这是否真正反映了"外生变量"场景？
2. 能否报告 "**仅处理内生变量 vs 完整 TimeXer**" 的性能差距？这才能量化外生变量的贡献。
3. 全局 token 的独立消融？移除全局 token 后性能变化？

### 对 DiffSTG 作者
1. 能否提供**不确定性校准指标**（可靠性图、期望校准误差 ECE）来补充 CRPS？
2. 仅 8 个采样样本是否足够？能否展示采样数对 CRPS 和校准的影响？
3. 与更强 deterministic 方法（如 AGCRN、D2STGNN）比较确定性预测结果？

---

## 五、建议的研究优先事项

基于以上分析，以下是该领域最紧迫的研究空白：

1. **统一外生框架**：需要一个能处理任意模态、任意数量的外生输入的理论框架，而非为每个场景定制设计
2. **真正的长期基准**：建立长期时空预测的统一基准（周/月/年尺度），并明确 horizon 定义
3. **严格的因果评估**：因果方法需要可验证的因果图、反事实实验或可干预的基准
4. **代码与可重复性标准化**：领域需要强制代码发布和标准化实验报告模板
5. **跨域 zero-shot 基准**：建立真正的跨域零样本评估协议（如交通→天气→能源）
6. **不确定性量化标准**：概率方法需要校准评估，而不仅仅是 CRPS
7. **LLM 能力的消解实验**：需要回答"LLM 提供了什么独特价值"——将 LLM 与更小、更便宜的替代品比较

---

*分析完成。所有引用均来自原始论文的文本提取版本。数字和引用在提取过程中尽力保持准确，建议交叉验证原文。*
