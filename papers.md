**使用LLM和多模态LLM进行时空预测与推理的研究论文搜索总结（2023-2026）**[[1]](https://openreview.net/forum?id=dFapOK8Rhb)[[2]](https://arxiv.org/html/2510.11282v1)

根据arXiv、WWW、KDD、AAAI、NeurIPS、ICML（2023-2026）搜索结果，该领域快速发展。LLM（GPT-4、LLaMA等）作为推理引擎处理文本外生信息（新闻、事件、报告），多模态LLM（GPT-4V、LLaVA、Gemini风格）融合卫星/交通图像、文本与时间序列；CoT/reflection、RAG、Agentic方法显著提升预测准确性、可解释性和零样本能力。UniST式的prompt engineering和ExoLLM式的文本-数值对齐是主流技术，Terra等数据集为基准提供支持。以下按查询点总结，并列出关键论文。[[3]](https://arxiv.org/html/2310.10196v3)

**1. LLMs作为时空预测的推理引擎**
GPT-4、LLaMA等通过Meta-task Instruction、多粒度提示（multi-grained prompts）和Dual TS-Text Attention处理文本外生信息（新闻、事件描述、报告），将自然语言知识转化为辅助信号指导数值/ST预测，避免仅依赖时间序列导致的虚假相关。**ExoLLM**是代表性工作，它让LLM理解动态外部影响并与TS对齐。其他工作结合reflection机制实现逐步推理，提升时空模式理解。[[4]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)[[5]](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869)

**2. Text-to-forecast管道**
从非结构化文本（气候报告、交通事件日志、流行病公告）中提取结构化信号（如事件嵌入、因果关系、语义特征），通过LLM生成特征向量供下游预测模型使用。典型管道包括LLM-based事件分析 + 反射/过滤 + 数值预测，常构建专用多模态数据集（如新闻-TS配对）。**GPT4MTS**和**From News to Forecast**等工作实现了高效端到端转换，提升复杂事件下的预测性能。[[6]](https://arxiv.org/html/2505.17637v2)

**3. 多模态LLM应用于时空数据**
**ST-Vision-LLM**、**Solar-VLM**、**GPT4MTS**等将卫星/交通图像视为视觉输入，与文本描述、时间序列一起处理。常用ViT视觉编码器 + 数值tokenization（单token浮点表示）+ cross-attention或两阶段对齐微调，实现联合推理。ST-Vision-LLM将交通矩阵视为图像序列喂给Vision-LLM，Solar-VLM特别针对太阳能/气象时空预测融合多源模态，接近GPT-4V/LLaVA在ST场景的扩展。[[7]](https://arxiv.org/html/2510.11282v2)[[8]](https://arxiv.org/html/2604.04145v1)

**4. Chain-of-Thought / 推理增强预测**
使用CoT prompting或reflection生成显式推理轨迹（如“事件X通过Y机制影响区域Z的流量”），然后注入预测头或作为上下文，提升准确性和可解释性。在数据稀疏ST场景中效果显著。**From News to Forecast**和TraffiCoT-R等工作将LLM推理痕迹与数值模型结合，COUNTS等进一步用RL优化CoT过程。[[9]](https://arxiv.org/pdf/2601.03248)

**5. RAG用于时空预测**
检索历史相似时空模式、外生知识图或相关事件文档，作为上下文增强LLM/TS基础模型的零样本预测。TS-RAG等框架检索历史pattern并生成augmented prompt，显著提升泛化能力，尤其在长时程ST预测中减少幻觉。常与Agent结合实现动态检索。[[3]](https://arxiv.org/html/2310.10196v3)

**6. 基于Agent的方法**
LLM Agent可查询数据库、运行模拟、过滤新闻、进行多轮reflection并合成时空预测。**From News to Forecast**使用生成式Agent处理事件；其他工作如Hierarchical LLM-Agent Framework（用于人类移动性/ST预测）和MACRO-LLM（多Agent协作下部分可观测ST环境）实现端到端模拟与预测。Agent能动态整合多源外生信息。[[10]](https://arxiv.org/html/2510.24802v1)

**7. 时间序列的Prompt Engineering**
将时空数据格式化为自然语言提示（如patch序列描述、知识引导prompt），或使用“Prompt-as-Hint”机制（可学习prompt网络 + 内存池存储时空知识如周期性、层次性、邻近性）。**UniST**和UrbanGPT是典型，通过prompt tuning实现零/少样本跨域ST预测，可轻松注入文本外生信息；Time-LLM式的reprogramming也将TS patch映射为LLM可理解的prompt。[[11]](https://arxiv.org/html/2402.11838v5)

**关键论文列表**（每篇包含标题、作者、年份、venue、1-2句贡献、URL/arXiv ID）：

- **标题**：Exploiting Language Power for Time Series Forecasting with Exogenous Variables (ExoLLM)。**作者**：Qihe Huang, Zhengyang Zhou, Kuo Yang, Yang Wang等。**年份**：2025。**venue**：WWW 2025。**贡献**：提出ExoLLM框架，利用Meta-task Instruction激活LLM的语言知识，通过多粒度提示捕捉新闻/事件等文本外生变量的多样化动态影响，并采用Dual TS-Text Attention实现对齐，有效提升时空/时间序列预测准确性，是LLM作为外生推理引擎的代表工作。**URL**：http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf 或 DOI: 10.1145/3696410.3714793。[[4]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)

- **标题**：GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting。**作者**：Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu等。**年份**：2024。**venue**：AAAI 2024。**贡献**：提出基于提示的框架，将数值时间序列与文本外生信息（报告、新闻）转化为结构化prompt输入GPT-like LLM，实现多模态联合预测，并构建相关数据集验证了文本到预测管道的有效性，提升了复杂事件下的准确性和可解释性。**URL**：https://ojs.aaai.org/index.php/AAAI/article/view/30383。[[12]](https://github.com/wpf535236337/LLMs4TS)

- **标题**：UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction。**作者**：Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li。**年份**：2024。**venue**：KDD 2024。**贡献**：提出UniST通用模型，通过知识引导的prompt learning（Prompt-as-Hint，使用时空记忆池生成邻近性、周期性等提示）实现城市时空多任务零/少样本预测，可方便融入文本外生信息，是prompt engineering在ST领域的典范。**URL**：https://arxiv.org/abs/2402.11838。[[11]](https://arxiv.org/html/2402.11838v5)

- **标题**：Vision-LLMs for Spatiotemporal Traffic Forecasting (ST-Vision-LLM)。**作者**：Ning Yang, Hengyu Zhong, Haijun Zhang, Randall Berry。**年份**：2025/2026。**venue**：arXiv:2510.11282。**贡献**：将时空交通预测重构为vision-language融合任务，使用Vision-LLM处理交通矩阵图像序列 + 高效数值tokenization和两阶段对齐微调，在长时预测和跨域少样本场景中显著优于基线，是多模态LLM（类似GPT-4V）处理图像+时间序列的代表工作。**URL**：https://arxiv.org/abs/2510.11282。[[2]](https://arxiv.org/html/2510.11282v1)

- **标题**：Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting。**作者**：Hang Fan, Haoran Pei等（相关团队）。**年份**：2026。**venue**：arXiv:2604.04145。**贡献**：提出多模态VLM框架，融合卫星图像、天气文本报告和数值时间序列，通过模态特定编码器与LLM交互，实现太阳能功率的时空增强预测，在多模态融合架构上为ST预测提供实用案例。**URL**：https://arxiv.org/abs/2604.04145。[[8]](https://arxiv.org/html/2604.04145v1)

- **标题**：UrbanGPT: Spatio-Temporal Large Language Models。**作者**：Zhonghang Li, Lianghao Xia, Chao Huang等。**年份**：2024。**venue**：KDD 2024 / arXiv:2403.00813。**贡献**：提出专为城市时空设计的LLM，结合ST依赖编码器和指令微调，将时空数据格式化为prompt，支持零样本多任务预测（如交通、移动性），有效整合文本外生知识。**URL**：https://arxiv.org/abs/2403.00813。[[13]](https://arxiv.org/html/2403.00813v3)

- **标题**：From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection。**作者**：Xinlei Wang等。**年份**：2024。**venue**：NeurIPS 2024。**贡献**：利用LLM + 生成式Agent对新闻/事件文本进行分析与reflection（CoT-style推理），提取结构化信号注入数值预测模型，提升了文本外生驱动的时空预测准确性与可解释性，是Agent-based与reasoning-augmented的典型结合。**URL**：相关arXiv或NeurIPS proceedings。

- **标题**：Spatio-Temporal LLM: Reasoning about Environments and Actions。**作者**：Haozhen Zheng等。**年份**：2025。**venue**：arXiv:2507.05258。**贡献**：提出多模态STLLM，融合点云/视频（类卫星图像）、文本与时空推理，引入专用数据集和基线模型，为多模态LLM在复杂时空环境下的推理提供了基础框架。**URL**：https://arxiv.org/abs/2507.05258。

此外，TS-RAG（NeurIPS 2025，检索增强零样本预测）、FSTLLM（ICML 2025，少样本ST LLM）和相关综述（如“Large Models for Time Series and Spatio-Temporal Data”更新版）也是重要参考。这些工作共同推动了从纯数值模型向LLM驱动、多模态、具身推理的时空预测范式转变。如需特定论文PDF细节或进一步扩展分析，可提供更多针对性查询。所有信息基于公开学术来源。


**时空基础模型处理外生多模态输入的研究论文搜索总结（2023-2026）**[[1]](https://arxiv.org/abs/2509.05779)[[2]](https://arxiv.org/abs/2505.17637)[[3]](https://neurips.cc/virtual/2024/poster/97768)

根据2023-2026年间arXiv、NeurIPS、ICLR、TKDE等来源的搜索结果，该领域快速发展，特别是Terra数据集的发布极大推动了多模态时空（ST）基础模型的研究。现有工作重点在于外生变量（exogenous variables，如数值协变量、文本、图像、知识图、物理约束）的建模、多模态融合架构、注入机制，以及将PINNs、神经算子、GNN与多模态外生输入结合。以下按查询点逐一总结，并列出关键论文。[[4]](https://github.com/LMissher/Awesome-Spatio-Temporal-Foundation-Models)[[5]](https://arxiv.org/html/2501.09045v2)

**1. 指定时空基础模型（UniST, Aurora, ClimaX, Pangu-Weather, GraphCast, UrbanDiT, TimeGPT, Lag-Llama, TimesFM, Moirai, Chronos）对外生变量的支持及方式**
- **Aurora**（2025，多模态TS基础模型）：全面支持文本（BERT嵌入）、图像（ViT特征，包括内生图像）和外生/领域知识，通过token distillation、Modality-Guided Self-Attention实现跨模态交互与零样本跨域生成式预测，支持原型引导流匹配。[[6]](https://arxiv.org/abs/2509.22295)
- **ClimaX**（2023）：将不同气候变量视为独立模态，使用variable tokenization在ViT上处理异构数据，支持多变量外生输入（如不同尺度、来源数据）。
- **Pangu-Weather、GraphCast**：支持多气象变量及静态外生（如地形、地海掩码、土壤类型），通常作为额外输入通道或掩码注入Transformer/GNN backbone。
- **UrbanDiT、UniST**：城市ST通用/开放世界基础模型，通过统一提示学习（prompt learning）或可学习提示内存池融入外生信息（如POI、文本描述、异构网格/图数据），支持零样本多任务。[[7]](https://arxiv.org/abs/2411.12164)
- **TimeGPT、Moirai、TimesFM、Lag-Llama、Chronos**：多数TS基础模型支持数值协变量（covariates/exogenous）。TimeGPT显式将历史/未来外生作为额外输入；Moirai/TimesFM支持动态协变量或XReg（残差回归）；Chronos（尤其是Chronos-2/ ChronosX扩展）原生支持过去/已知未来协变量（实值/类别）；Lag-Llama侧重滞后与统计特征作为外生。整体趋势是从纯单变量向协变量/多模态条件化演进。[[8]](https://arxiv.org/pdf/2501.09045)

**2. 时空数据的多模态融合架构**
常见架构包括：**cross-attention**（时间序列+文本+图像间交互，如E²-CSTP和Aurora的modality-guided self-attention）；**early fusion**（输入级拼接特征）、**intermediate fusion**（latent space跨模态交互，最常见于Transformer）、**late fusion**（决策级融合）；**mixture-of-experts (MoE)** 用于模态路由（如ExoST的latent space gated expert模块，动态选择显著信号）；**gated fusion**（门控机制平衡不同模态/历史 vs 未来影响）。这些在ST场景中常与GNN、Mamba或Diffusion Transformer结合，提升对异构时空依赖的建模。[[2]](https://arxiv.org/abs/2505.17637)

**3. 外生信息（文本嵌入、ViT图像特征、知识图、物理方程）的编码与注入方式**
- **文本**：使用BERT/LLM（如GPT-4生成描述）得到嵌入，投影到模型维度后通过cross-attention、addition或prompt注入ST backbone。
- **图像**：ViT/CNN提取patch/token特征，常与TS token一起进行token distillation或modality-guided attention融合。
- **知识图（KG）**：GNN嵌入实体/关系，作为节点特征或额外图结构与ST-GNN融合。
- **物理方程**：直接纳入损失函数（PINN风格）或作为硬约束/内核融入神经算子（FNO/DeepONet）。
注入常见方式：输入级concat、latent space gated expert/MoE动态路由、context-aware weighting平衡历史/未来、adapter或条件化机制。ExoST的“select then balance”范式是典型代表，能与现有ST backbone即插即用。[[1]](https://arxiv.org/abs/2509.05779)

**4. PINNs和神经算子（FNO, DeepONet）向多模态时空设置的扩展**
PINNs通过多模态输入（数值+图像+文本）与物理先验结合，提升可解释性与泛化；神经算子扩展通过额外模态分支、融合模块或多算子学习处理异构数据。代表性工作包括PI-MFM（将物理方程直接纳入预训练与适配的多模态基础模型，用于PDE求解）和多模态PINN（融合图像特征与辐射传输方程等）。这些扩展常用于气候、城市环境、结构分析等ST场景，与Terra数据集结合潜力巨大。[[9]](https://arxiv.org/abs/2512.23056)

**5. 图神经网络（STGCN, GraphWaveNet等）结合多模态外生输入的调研或近期论文**
STGCN、GraphWaveNet等经典ST-GNN常通过特征拼接、单独编码器+cross-attention/gating融合外生（天气数值、POI文本BERT嵌入、卫星图像ViT特征、KG）。近期工作多见于城市/交通领域（如E²-CSTP中GCN+Mamba与多模态因果融合），或physics-informed ST-GNN。STFM pipeline综述中涵盖graph-based STFMs的多模态扩展，虽无单一全面survey，但Awesome列表和多模态城市计算相关综述提供了丰富参考。常见挑战是模态不平衡与时空-图对齐。[[8]](https://arxiv.org/pdf/2501.09045)

**6. 关键论文列表**（每篇包含标题、作者、年份、venue、1-2句贡献、URL/arXiv ID）
以下聚焦查询中指定的关键论文及高度相关的2023-2026工作（基于arXiv、NeurIPS、ICLR、TKDE等）。[[6]](https://arxiv.org/abs/2509.22295)[[7]](https://arxiv.org/abs/2411.12164)

- **标题**：Select, then Balance: Exploring Exogenous Variable Modeling of Spatio-Temporal Forecasting（ExoST）。**作者**：Wei Chen, Yuqian Wu, Yuanshao Zhu, Xixuan Hao, Shiyu Wang, Xiaofang Zhou, Yuxuan Liang。**年份**：2025。**venue**：arXiv:2509.05779。**贡献**：首次系统探讨时空预测中外生变量建模的挑战（不一致效应与历史/未来不平衡），提出ExoST即插即用框架，采用“select then balance”范式，使用latent space gated expert模块动态选择融合外生信号，并以siamese dual-branch + context-aware weighting实现平衡，与现有ST backbone高度兼容，实验验证其通用性、鲁棒性和效率。**URL**：https://arxiv.org/abs/2509.05779。[[1]](https://arxiv.org/abs/2509.05779)

- **标题**：Causal Spatio-Temporal Prediction: An Effective and Efficient Multi-Modal Approach（E²-CSTP / E-CSTP）。**作者**：Yuting Huang, Ziquan Fang, Zhihao Zeng, Lu Chen, Yunjun Gao。**年份**：2025。**venue**：arXiv:2505.17637。**贡献**：针对多模态时空预测中的融合不足、混杂偏差和计算复杂度问题，提出E²-CSTP框架，使用cross-modal attention和gating机制进行多模态融合，双分支因果推理（主分支预测、辅助分支建模额外模态并施加因果干预消除偏差），结合GCN+Mamba高效编码，在4个真实数据集上显著优于9个SOTA方法，同时降低计算开销。**URL**：https://arxiv.org/abs/2505.17637。[[2]](https://arxiv.org/abs/2505.17637)

- **标题**：Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth。**作者**：Wei Chen, Xixuan Hao, Yuankai Wu, Yuxuan Liang。**年份**：2024。**venue**：NeurIPS 2024 (Datasets and Benchmarks Track)。**贡献**：发布大规模多模态地球时空数据集，包含648万全球网格45+年小时级气象时间序列，以及对应地理图像和LLM生成的解释性文本作为空间补充信息，为训练支持多模态外生输入的ST基础模型提供了关键基准，推动空间智能研究。**URL**：https://proceedings.neurips.cc/paper_files/paper/2024/file/7a6a7fbd1ee0c9684b3f919f79d129ef-Paper-Datasets_and_Benchmarks_Track.pdf (或GitHub: https://github.com/CityMind-Lab/NeurIPS24-Terra)。[[3]](https://neurips.cc/virtual/2024/poster/97768)

- **标题**：Aurora: Towards Universal Generative Multimodal Time Series Forecasting。**作者**：Xingjian Wu, Jianxin Jin, Wanghui Qiu, Peng Chen, Yang Shu, Bin Yang, Chenjuan Guo。**年份**：2025（ICLR 2026提交）。**venue**：arXiv:2509.22295。**贡献**：提出首个多模态时间序列基础模型Aurora，支持文本（BERT）、图像（ViT）等外生多模态输入，在跨域多模态语料上预训练，通过token distillation、Modality-Guided Self-Attention实现跨模态知识提取与零样本生成式预测，并使用Prototype-Guided Flow Matching提升性能，展现强大跨域泛化能力。**URL**：https://arxiv.org/abs/2509.22295。[[6]](https://arxiv.org/abs/2509.22295)

- **标题**：Diffusion Transformers as Open-World Spatiotemporal Foundation Models（UrbanDiT）。**作者**：Yuan Yuan et al.（含Yuxuan Liang等）。**年份**：2024。**venue**：arXiv:2411.12164。**贡献**：提出开放世界城市时空基础模型UrbanDiT，基于Diffusion Transformer和统一提示学习框架，将网格/图数据统一为序列，支持多任务（双向预测、补全、外推）、零样本泛化和异构数据适应，可通过prompt融入多模态外生信息。**URL**：https://arxiv.org/abs/2411.12164。[[7]](https://arxiv.org/abs/2411.12164)

- **标题**：Physics-informed multimodal foundation model for solving partial differential equations（PI-MFM）。**作者**：M. Zhu, J. Sun, Z. Zhang, H. Schaeffer, L. Lu等。**年份**：2025。**venue**：arXiv:2512.23056。**贡献**：提出物理信息多模态基础模型框架，在预训练和适配阶段直接强制执行 governing equations，支持多模态输入扩展到PDE求解，是PINNs与神经算子（FNO/DeepONet风格）向多模态时空设置的重要扩展。**URL**：https://arxiv.org/abs/2512.23056。[[9]](https://arxiv.org/abs/2512.23056)

- **标题**：Spatio-Temporal Foundation Models: Vision, Challenges, and Opportunities。**作者**：Bryan Hooi, Adam Goodge, Wee Siong Ng, See-Kiong Ng等。**年份**：2025。**venue**：arXiv:2501.09045。**贡献**：系统阐述ST基础模型的愿景、当前碎片化状态（交通 vs 天气模型）、泛化能力差距，并重点讨论多模态训练（融合TS、影像、文本）、外生整合、因果推理等机遇，为该领域提供路线图，包括对ClimaX、Pangu、UniST等模型的评估。**URL**：https://arxiv.org/abs/2501.09045。[[5]](https://arxiv.org/html/2501.09045v2)

- **标题**：Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review。**作者**：Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng等。**年份**：2025/2026。**venue**：arXiv (后续TKDE 2026)。**贡献**：从pipeline视角（数据、预训练、迁移等）全面综述ST基础模型，详细讨论多模态整合、外生变量处理、UniST/UrbanDiT/ClimaX/Moirai/Chronos等模型，支持方式及未来融合架构（cross-attention、MoE、gated）方向，是领域重要参考文献。**URL**：https://arxiv.org/abs/2501.09045 或相关版本（如2506.01364）。[[4]](https://github.com/LMissher/Awesome-Spatio-Temporal-Foundation-Models)

- **标题**：A Multimodal Physics-Informed Neural Network Approach for Mean Radiant Temperature Modeling（多模态PINN示例）。**作者**：相关城市气候研究者（2025 arXiv）。**年份**：2025。**venue**：arXiv:2503.08482。**贡献**：提出多模态PINN，融合数值环境特征、鱼眼图像（ResNet提取）与辐射传输物理方程到损失函数，实现准确、可解释的城市平均辐射温度建模，是PINNs向多模态ST时空扩展的实用案例。**URL**：https://arxiv.org/abs/2503.08482。

此外，ClimaX（arXiv:2301.10343, 2023, ICLR/ICML相关）、UniST（arXiv:2402.11838, KDD 2024）也是重要基线。搜索显示2025-2026年多模态ST FM论文数量快速增长，Terra数据集已成为常见基准。未来方向包括更强的MoE路由、因果多模态融合及物理-数据联合预训练。

这些论文和综述为进一步研究提供了坚实基础。如需特定论文PDF细节或更多扩展，可进一步探索。所有信息基于公开arXiv和会议记录。




**使用LLM和多模态LLM进行时空预测与推理的研究论文搜索总结（2023-2026）**[[1]](https://openreview.net/forum?id=dFapOK8Rhb)[[2]](https://arxiv.org/html/2510.11282v1)

根据arXiv、WWW、KDD、AAAI、NeurIPS、ICML（2023-2026）搜索结果，该领域快速发展。LLM（GPT-4、LLaMA等）作为推理引擎处理文本外生信息（新闻、事件、报告），多模态LLM（GPT-4V、LLaVA、Gemini风格）融合卫星/交通图像、文本与时间序列；CoT/reflection、RAG、Agentic方法显著提升预测准确性、可解释性和零样本能力。UniST式的prompt engineering和ExoLLM式的文本-数值对齐是主流技术，Terra等数据集为基准提供支持。以下按查询点总结，并列出关键论文。[[3]](https://arxiv.org/html/2310.10196v3)

**1. LLMs作为时空预测的推理引擎**
GPT-4、LLaMA等通过Meta-task Instruction、多粒度提示（multi-grained prompts）和Dual TS-Text Attention处理文本外生信息（新闻、事件描述、报告），将自然语言知识转化为辅助信号指导数值/ST预测，避免仅依赖时间序列导致的虚假相关。**ExoLLM**是代表性工作，它让LLM理解动态外部影响并与TS对齐。其他工作结合reflection机制实现逐步推理，提升时空模式理解。[[4]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)[[5]](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869)

**2. Text-to-forecast管道**
从非结构化文本（气候报告、交通事件日志、流行病公告）中提取结构化信号（如事件嵌入、因果关系、语义特征），通过LLM生成特征向量供下游预测模型使用。典型管道包括LLM-based事件分析 + 反射/过滤 + 数值预测，常构建专用多模态数据集（如新闻-TS配对）。**GPT4MTS**和**From News to Forecast**等工作实现了高效端到端转换，提升复杂事件下的预测性能。[[6]](https://arxiv.org/html/2505.17637v2)

**3. 多模态LLM应用于时空数据**
**ST-Vision-LLM**、**Solar-VLM**、**GPT4MTS**等将卫星/交通图像视为视觉输入，与文本描述、时间序列一起处理。常用ViT视觉编码器 + 数值tokenization（单token浮点表示）+ cross-attention或两阶段对齐微调，实现联合推理。ST-Vision-LLM将交通矩阵视为图像序列喂给Vision-LLM，Solar-VLM特别针对太阳能/气象时空预测融合多源模态，接近GPT-4V/LLaVA在ST场景的扩展。[[7]](https://arxiv.org/html/2510.11282v2)[[8]](https://arxiv.org/html/2604.04145v1)

**4. Chain-of-Thought / 推理增强预测**
使用CoT prompting或reflection生成显式推理轨迹（如“事件X通过Y机制影响区域Z的流量”），然后注入预测头或作为上下文，提升准确性和可解释性。在数据稀疏ST场景中效果显著。**From News to Forecast**和TraffiCoT-R等工作将LLM推理痕迹与数值模型结合，COUNTS等进一步用RL优化CoT过程。[[9]](https://arxiv.org/pdf/2601.03248)

**5. RAG用于时空预测**
检索历史相似时空模式、外生知识图或相关事件文档，作为上下文增强LLM/TS基础模型的零样本预测。TS-RAG等框架检索历史pattern并生成augmented prompt，显著提升泛化能力，尤其在长时程ST预测中减少幻觉。常与Agent结合实现动态检索。[[3]](https://arxiv.org/html/2310.10196v3)

**6. 基于Agent的方法**
LLM Agent可查询数据库、运行模拟、过滤新闻、进行多轮reflection并合成时空预测。**From News to Forecast**使用生成式Agent处理事件；其他工作如Hierarchical LLM-Agent Framework（用于人类移动性/ST预测）和MACRO-LLM（多Agent协作下部分可观测ST环境）实现端到端模拟与预测。Agent能动态整合多源外生信息。[[10]](https://arxiv.org/html/2510.24802v1)

**7. 时间序列的Prompt Engineering**
将时空数据格式化为自然语言提示（如patch序列描述、知识引导prompt），或使用“Prompt-as-Hint”机制（可学习prompt网络 + 内存池存储时空知识如周期性、层次性、邻近性）。**UniST**和UrbanGPT是典型，通过prompt tuning实现零/少样本跨域ST预测，可轻松注入文本外生信息；Time-LLM式的reprogramming也将TS patch映射为LLM可理解的prompt。[[11]](https://arxiv.org/html/2402.11838v5)

**关键论文列表**（每篇包含标题、作者、年份、venue、1-2句贡献、URL/arXiv ID）：

- **标题**：Exploiting Language Power for Time Series Forecasting with Exogenous Variables (ExoLLM)。**作者**：Qihe Huang, Zhengyang Zhou, Kuo Yang, Yang Wang等。**年份**：2025。**venue**：WWW 2025。**贡献**：提出ExoLLM框架，利用Meta-task Instruction激活LLM的语言知识，通过多粒度提示捕捉新闻/事件等文本外生变量的多样化动态影响，并采用Dual TS-Text Attention实现对齐，有效提升时空/时间序列预测准确性，是LLM作为外生推理引擎的代表工作。**URL**：http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf 或 DOI: 10.1145/3696410.3714793。[[4]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)

- **标题**：GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting。**作者**：Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu等。**年份**：2024。**venue**：AAAI 2024。**贡献**：提出基于提示的框架，将数值时间序列与文本外生信息（报告、新闻）转化为结构化prompt输入GPT-like LLM，实现多模态联合预测，并构建相关数据集验证了文本到预测管道的有效性，提升了复杂事件下的准确性和可解释性。**URL**：https://ojs.aaai.org/index.php/AAAI/article/view/30383。[[12]](https://github.com/wpf535236337/LLMs4TS)

- **标题**：UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction。**作者**：Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li。**年份**：2024。**venue**：KDD 2024。**贡献**：提出UniST通用模型，通过知识引导的prompt learning（Prompt-as-Hint，使用时空记忆池生成邻近性、周期性等提示）实现城市时空多任务零/少样本预测，可方便融入文本外生信息，是prompt engineering在ST领域的典范。**URL**：https://arxiv.org/abs/2402.11838。[[11]](https://arxiv.org/html/2402.11838v5)

- **标题**：Vision-LLMs for Spatiotemporal Traffic Forecasting (ST-Vision-LLM)。**作者**：Ning Yang, Hengyu Zhong, Haijun Zhang, Randall Berry。**年份**：2025/2026。**venue**：arXiv:2510.11282。**贡献**：将时空交通预测重构为vision-language融合任务，使用Vision-LLM处理交通矩阵图像序列 + 高效数值tokenization和两阶段对齐微调，在长时预测和跨域少样本场景中显著优于基线，是多模态LLM（类似GPT-4V）处理图像+时间序列的代表工作。**URL**：https://arxiv.org/abs/2510.11282。[[2]](https://arxiv.org/html/2510.11282v1)

- **标题**：Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting。**作者**：Hang Fan, Haoran Pei等（相关团队）。**年份**：2026。**venue**：arXiv:2604.04145。**贡献**：提出多模态VLM框架，融合卫星图像、天气文本报告和数值时间序列，通过模态特定编码器与LLM交互，实现太阳能功率的时空增强预测，在多模态融合架构上为ST预测提供实用案例。**URL**：https://arxiv.org/abs/2604.04145。[[8]](https://arxiv.org/html/2604.04145v1)

- **标题**：UrbanGPT: Spatio-Temporal Large Language Models。**作者**：Zhonghang Li, Lianghao Xia, Chao Huang等。**年份**：2024。**venue**：KDD 2024 / arXiv:2403.00813。**贡献**：提出专为城市时空设计的LLM，结合ST依赖编码器和指令微调，将时空数据格式化为prompt，支持零样本多任务预测（如交通、移动性），有效整合文本外生知识。**URL**：https://arxiv.org/abs/2403.00813。[[13]](https://arxiv.org/html/2403.00813v3)

- **标题**：From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection。**作者**：Xinlei Wang等。**年份**：2024。**venue**：NeurIPS 2024。**贡献**：利用LLM + 生成式Agent对新闻/事件文本进行分析与reflection（CoT-style推理），提取结构化信号注入数值预测模型，提升了文本外生驱动的时空预测准确性与可解释性，是Agent-based与reasoning-augmented的典型结合。**URL**：相关arXiv或NeurIPS proceedings。

- **标题**：Spatio-Temporal LLM: Reasoning about Environments and Actions。**作者**：Haozhen Zheng等。**年份**：2025。**venue**：arXiv:2507.05258。**贡献**：提出多模态STLLM，融合点云/视频（类卫星图像）、文本与时空推理，引入专用数据集和基线模型，为多模态LLM在复杂时空环境下的推理提供了基础框架。**URL**：https://arxiv.org/abs/2507.05258。

此外，TS-RAG（NeurIPS 2025，检索增强零样本预测）、FSTLLM（ICML 2025，少样本ST LLM）和相关综述（如“Large Models for Time Series and Spatio-Temporal Data”更新版）也是重要参考。这些工作共同推动了从纯数值模型向LLM驱动、多模态、具身推理的时空预测范式转变。如需特定论文PDF细节或进一步扩展分析，可提供更多针对性查询。所有信息基于公开学术来源。

**以下是针对2023-2026年间使用LLMs和多模态LLMs进行时空预测（spatiotemporal prediction）和推理（reasoning）相关研究的精选论文列表。** 我通过网络搜索和论文页面浏览，优先选择了查询中提到的关键论文（ExoLLM、GPT4MTS、ST-Vision-LLM、Solar-VLM、UniST等），并补充了覆盖其他子主题（如CoT/reasoning-augmented、RAG、agent-based、text-to-forecast管道、prompt engineering）的代表性工作。[[1]](https://openreview.net/forum?id=dFapOK8Rhb)[[2]](https://ojs.aaai.org/index.php/AAAI/article/view/30383)[[3]](https://arxiv.org/abs/2510.11282)

这些论文主要聚焦于：LLM作为reasoning engine处理文本外生信息（新闻、报告、事件）；从非结构化文本提取结构化信号；多模态融合（卫星影像/视觉表示 + 文本 + 时间序列）；CoT/推理增强预测；RAG检索历史模式或知识；LLM代理查询数据库、模拟并合成预测；以及将时空数据格式化为文本prompt的技术（如Prompt-as-Hint或reprogramming）。

我为每个论文提供了：**标题、作者、年份、venue、1-2句贡献描述、URL/arXiv ID**。列表按查询类别大致分组，部分论文跨多个类别。所有工作均在指定时间范围内，聚焦时空/时间序列预测与解释性提升。

### 1. LLMs as reasoning engines for spatiotemporal forecasting（处理文本外生信息如新闻/报告来辅助数值预测）
- **Title**: Exploiting Language Power for Time Series Forecasting with Exogenous Variables (ExoLLM)
  **Authors**: Qihe Huang, Zhengyang Zhou, Kuo Yang, Yang Wang
  **Year**: 2025
  **Venue**: WWW 2025
  **Contribution**: 提出ExoLLM框架（首个LLM驱动的FEV方法），通过Meta-task Instruction激活LLM从NLP到时间序列外生变量预测的知识转移，使用Multi-grained Prompts捕捉新闻/报告等外生信息的自然属性、趋势相关性和周期关系，并借助Dual TS-Text Attention对齐文本-数值特征空间，让GPT-4/LLaMA等LLM作为reasoning engine有效利用动态文本外生知识提升数值/时空预测，避免虚假相关。[[4]](https://dl.acm.org/doi/10.1145/3696410.3714793)[[5]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)
  **URL/arXiv**: https://dl.acm.org/doi/10.1145/3696410.3714793 或 PDF: http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf

- **Title**: From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection
  **Authors**: Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, Junhua Zhao
  **Year**: 2024
  **Venue**: NeurIPS 2024
  **Contribution**: 利用LLM生成式代理迭代过滤新闻/事件文本（非结构化外生信息），结合反射（reflection）和类CoT的人类式推理评估其对时间序列的影响，并与数值数据融合微调LLM进行预测；显著提升了包含时空场景（如电力、交通）的预测准确性和可解释性，是典型的text-to-forecast与reasoning-augmented管道。[[6]](https://neurips.cc/virtual/2024/poster/93316)
  **URL/arXiv**: https://arxiv.org/abs/2409.17515 (或NeurIPS proceedings)

### 2. Text-to-forecast pipelines（从气候报告、交通事件日志、流行病学公告等非结构化文本中提取结构化信号）
（上述“From News to Forecast”高度相关，同时覆盖文本提取与下游预测。GPT4MTS也直接支持文本信号提取。）

- **Title**: GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting
  **Authors**: Furong Jia, Kevin Wang, Yuxuan Zheng, Donghui Cao, Yun Liu
  **Year**: 2024
  **Venue**: AAAI 2024
  **Contribution**: 提出prompt-based LLM框架（兼容GPT-4等），同时处理数值时间序列和文本 exogenous信息（如新闻、报告、公告），有效从非结构化文本中提取结构化信号用于下游预测，在多模态数据集（如GDELT）上显著优于纯数值或早期LLM基线。[[2]](https://ojs.aaai.org/index.php/AAAI/article/view/30383)
  **URL/arXiv**: https://ojs.aaai.org/index.php/AAAI/article/view/30383 (相关arXiv搜索可用)

### 3. Multimodal LLMs applied to spatiotemporal data（处理卫星影像 + 文本 + 时间序列，如GPT-4V、Gemini、LLaVA风格）
- **Title**: Vision-LLMs for Spatiotemporal Traffic Forecasting (ST-Vision-LLM)
  **Authors**: Ning Yang, Hengyu Zhong, Haijun Zhang, Randall Berry
  **Year**: 2025
  **Venue**: arXiv:2510.11282 (可能后续会议)
  **Contribution**: 将时空交通预测重构为vision-language融合任务，使用Vision-LLMs处理交通矩阵的图像化表示（类似卫星/视觉数据）、文本和时间序列，通过高效数值tokenization和两阶段微调（SFT+RL），在长期预测和跨域few-shot场景中大幅超越现有方法（长时预测提升约15.6%），适用于卫星影像等时空多模态数据。[[3]](https://arxiv.org/abs/2510.11282)
  **URL/arXiv**: https://arxiv.org/abs/2510.11282

- **Title**: Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting
  **Authors**: Hang Fan, Haoran Pei, Runze Liang, Weican Liu, Long Cheng, Wei Wei
  **Year**: 2026
  **Venue**: arXiv:2604.04145
  **Contribution**: 提出统一的多模态VLM框架，通过模态特定编码器融合卫星影像（视觉）、文本天气报告和时间序列数据，并使用图注意力网络与跨站点注意力建模复杂时空依赖，实现光伏/太阳能功率的增强预测，是卫星影像+文本+TS多模态LLM的典型应用。[[7]](https://arxiv.org/html/2604.04145v1)
  **URL/arXiv**: https://arxiv.org/abs/2604.04145

### 4. Chain-of-thought / reasoning-augmented forecasting（使用LLM推理轨迹提升准确性和可解释性）
（“From News to Forecast”中的reflection/CoT机制高度相关。另见COUNTS等RL训练CoT的工作，以及下面Time-LLM的prompt reasoning。）

### 5. Retrieval-augmented generation (RAG) for spatiotemporal prediction
- **Title**: Retrieval Augmented Time Series Forecasting (RAF)
  **Authors**: Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz, Samet Oymak
  **Year**: 2024
  **Venue**: arXiv:2411.08249 (相关ICML/NeurIPS工作)
  **Contribution**: 提出原则性的RAG框架（RAF），为LLM/时间序列基础模型检索相关历史模式、相似序列或外生知识来增强prompt/context，显著提升零样本预测能力和泛化性，可自然扩展到时空预测中处理动态事件驱动数据。[[8]](https://arxiv.org/abs/2411.08249)
  **URL/arXiv**: https://arxiv.org/abs/2411.08249
  （相关：TS-RAG，NeurIPS 2025，检索语义相关TS模式增强零样本预测。）

### 6. Agent-based approaches（LLM代理查询数据库、运行模拟、合成时空预测）
（“From News to Forecast”明确使用generative agents进行迭代过滤、推理和合成。另有DCATS/FLAIRR-TS等LLM-agent工作：如Visa Research的“Empowering Time Series Forecasting with LLM-Agents”（arXiv:2508.04231，2025），代理进行数据精炼、数据库查询和模拟优化；FLAIRR-TS使用迭代精炼+RAG的forecasting agent。）

### 7. Prompt engineering for time series（将时空数据格式化为文本prompt，如“Prompt-as-Hint”）
- **Title**: UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction
  **Authors**: Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li
  **Year**: 2024
  **Venue**: KDD 2024
  **Contribution**: 提出prompt-empowered通用模型，通过大规模城市时空数据预训练（掩码策略捕捉动态）和时空知识引导的prompt tuning（提取邻近性、层次、周期性等作为“Prompt-as-Hint”），实现跨数据集/城市的零样本/少样本时空预测泛化，体现了有效的时空数据文本格式化技术。[[9]](https://arxiv.org/html/2402.11838v5)
  **URL/arXiv**: https://arxiv.org/abs/2402.11838

**补充推荐基础/相关工作**（prompt & CoT）：
- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models** (Ming Jin et al., ICLR 2024, arXiv:2310.01728)。通过reprogramming将TS patch转为文本原型 + Prompt-as-Prefix (PaP)注入领域知识，实现LLM的零样本/少样本时空预测与推理增强。高度契合prompt engineering和reasoning-augmented类别。

**总体观察**：这些工作显示LLM（尤其是GPT-4系列、LLaMA、Vision-LLMs如LLaVA/Gemini风格）在时空领域的趋势是从纯数值转向多模态（视觉+文本+TS）、agentic reasoning和RAG增强。ExoLLM和From News to Forecast直接体现了用新闻/事件文本辅助预测；多模态论文处理卫星/影像数据；UniST和Time-LLM突出prompt设计；RAG/agent论文提升可解释性和动态适应性。更多领域特定工作（如流行病学公告提取或气候报告）可见相关survey（如“Empowering Time Series Analysis with LLMs”, IJCAI 2024）。如果需要更多论文全文细节、特定领域扩展（如epidemiology）或2026最新工作，我可以进一步搜索或浏览具体PDF。[[10]](https://raw.githubusercontent.com/mlresearch/v235/main/assets/jin24i/jin24i.pdf)

（信息主要来源于arXiv、conference proceedings、Semantic Scholar和GitHub仓库，日期截至当前2026年7月。）


**1. 误差累积在自回归时空预测中的问题——多模态外生信号如何帮助纠正长时漂移**

- **Title**: Select, then Balance: Exploring Exogenous Variable Modeling of Spatio-Temporal Forecasting (ExoST框架)
  **Authors**: Wei Chen, Yuqian Wu, Yuanshao Zhu, Xixuan Hao, Shiyu Wang, Xiaofang Zhou, Yuxuan Liang
  **Year**: 2025 (arXiv v1 2025.9，修订至2026)
  **Venue**: arXiv:2509.05779 (cs.LG)
  **Contribution**: 首次系统研究时空预测中外生变量建模的两大核心挑战（不同变量影响不一致、历史与未来数据不平衡），提出即插即用的ExoST框架，采用“select then balance”范式，通过潜在空间门控专家模块动态选择并重构显著外生信号，再用孪生双分支骨干网络与上下文感知加权实现动态平衡，有效缓解自回归长时漂移；实验显示外生信息对较长horizon（周至季节）的益处更显著。[[1]](https://arxiv.org/html/2509.05779v2)[[2]](https://arxiv.org/abs/2509.05779)

- **Title**: ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction
  **Authors**: Kshitij Tayal, Arvind Renganathan 等
  **Year**: 2024
  **Venue**: arXiv:2410.12184 (ICDM 2024相关)
  **Contribution**: 将过去与当前/预测外生变量视为不同模态，使用跨时间融合模块（cross-temporal fusion）与Transformer编码器-解码器整合到自回归预测中，对比PatchTST、iTransformer、DLinear等，显著减少长horizon误差累积和漂移，尤其在噪声预测的外生场景下表现稳健；可扩展至时空多模态输入。[[3]](https://arxiv.org/html/2410.12184v1)[[4]](https://arxiv.org/abs/2410.12184)

**2. 长序列架构（Informer, Autoformer, FEDformer, PatchTST, TimesNet）——对外生输入的处理及在极长horizon（周至季节）的扩展性**

- **Title**: Spatio-Temporal Forecasting: A Survey of Data-Driven Models Using Exogenous Data
  **Authors**: Safaa Berkani, Bassma Guermah, Mehdi Zakroum, Mounir Ghogho
  **Year**: 2023
  **Venue**: IEEE Access
  **Contribution**: 系统综述使用外生数据的时空预测数据驱动模型，提供融合阶段分类学（早期/中期/晚期融合），讨论Transformer类模型（如Informer/Autoformer分解、FEDformer频率增强、PatchTST patching、TimesNet多周期）如何整合天气、事件等外生信号，并分析其在长序列上的局限与扩展性；为后续工作提供基准，包括误差累积和多尺度问题。[[5]](https://www.researchgate.net/publication/371268810_Spatio-Temporal_Forecasting_a_Survey_of_Data-Driven_Models_using_Exogenous_Data)[[6]](https://www.semanticscholar.org/paper/Spatio-Temporal-Forecasting%3A-A-Survey-of-Models-Berkani-Guermah/279278eb33bda2adf7e2b938fcc58ba87cd97203)

- **Title**: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (及相关PTSTG/Patch-based ST扩展)
  **Authors**: Yuxuan Wang, Haixu Wu, Jiaxiang Dong 等（相关PTSTG工作见2025 Applied Sciences）
  **Year**: 2024
  **Venue**: NeurIPS 2024, arXiv:2402.19072
  **Contribution**: 在PatchTST等基础上设计patch-wise自注意力（内生）+ variate-wise跨注意力（外生），有效注入多模态外生变量，减少长horizon自回归漂移；基准测试显示其在长序列（包括周级）上优于Informer/Autoformer/FEDformer家族，扩展性强，可用于时空图设置。[[7]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)

**3. 状态空间模型（Mamba, S4, S5）用于长程时空依赖与多模态条件化**

- **Title**: STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model
  **Authors**: Lincan Li, Hanchen Wang, Wenjie Zhang, Adelle Coster
  **Year**: 2024
  **Venue**: arXiv:2403.12418
  **Contribution**: 首个将选择性状态空间模型（Mamba/SSSM）应用于时空图（STG）预测，将网络视为系统，使用Spatial-Temporal Selective State Space Module (ST-S3M)精准聚焦潜在特征，并结合Kalman Filtering GNN (KFGN)融合多粒度嵌入；线性复杂度下高效捕捉长程时空依赖，可通过节点特征或外部输入实现多模态条件化，在交通等基准上超越GNN基线。[[8]](https://arxiv.org/abs/2403.12418)

- **Title**: Multi-scale global supply chain spatiotemporal demand forecasting based on Mamba SSM (及DSTGA-Mamba/STCM-Mamba等变体)
  **Authors**: L. Gao 等（相关DSTGA-Mamba: L. Chen 等, 2025）
  **Year**: 2026
  **Venue**: Discover Artificial Intelligence (Springer)
  **Contribution**: 以Mamba SSM为主干，结合动态图注意力与事件感知嵌入，实现多尺度层次化建模，支持不同尺度外生影响（事件/宏观）；有效捕捉长程依赖与多模态条件化，适用于季节级长时预测，计算效率高。[[9]](https://link.springer.com/article/10.1007/s44163-026-01053-1)

**4. Neural ODE / 连续时间模型用于不规则采样时空数据与外生事件流**

- **Title**: RainODE: Continuous-Time Precipitation Forecasting with Latent Neural ODEs (及Spatio-Temporal Hypergraph Neural ODE Network for Traffic Forecasting, 2023)
  **Authors**: 相关工作包括RainODE作者组及C. Yao 等（hypergraph Neural ODE）
  **Year**: 2026 (RainODE arXiv ~2606) / 2023 (hypergraph变体)
  **Venue**: arXiv
  **Contribution**: 将precipitation预测重构为连续时间动力系统，在潜在空间中使用Neural ODE建模导数一致的动态，而非离散帧映射；适合不规则采样时空数据，可整合外生事件流作为连续演化驱动，改善长时预测稳定性。类似Temporal Graph Neural ODE工作处理不规则时空轨迹与事件。[[10]](https://arxiv.org/html/2606.29855v1)[[11]](https://github.com/Emory-Melody/Awesome-Graph-Neural-Differential-Equations)

**5. 多尺度/层次化时间建模——捕捉短时动态与长时趋势，以及不同尺度外生影响**

- **Title**: STDNet: A Spatio-Temporal Decomposition Neural Network for Multivariate Time Series Forecasting (及上述Multi-scale Mamba 2026工作)
  **Authors**: Z. Jiang 等（Multi-scale Mamba为L. Gao等）
  **Year**: 2024 (STDNet) / 2026
  **Venue**: 相关期刊 / Springer
  **Contribution**: 结合时空分解（受Autoformer启发）与双残差堆叠的MLP模型，同时捕捉多尺度趋势与季节性，外生影响可在不同分解层级注入；Multi-scale Mamba变体进一步用层次Mamba+动态图实现短时动态、长时趋势与多尺度外生事件的统一建模。[[12]](https://www.sciopen.com/article/10.26599/TST.2023.9010105)[[9]](https://link.springer.com/article/10.1007/s44163-026-01053-1)

**6. 不确定性量化（Bayesian NN、deep ensembles、conformal prediction、probabilistic/diffusion-based）与多模态输入——外生信息如何减少aleatoric vs epistemic uncertainty**

- **Title**: DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models (及DYffusion, BayesNF等)
  **Authors**: Haomin Wen 等（DYffusion: Salva Rühling Cachay, Rose Yu等, 2023 NeurIPS；BayesNF ~2024）
  **Year**: 2023
  **Venue**: SIGSPATIAL 2023 / NeurIPS
  **Contribution**: 将STGNN与去噪扩散模型结合，实现概率时空图预测，提供良好校准的不确定性量化；扩散过程可条件于多模态外生输入，更好建模aleatoric不确定性（数据内在变异性），并通过外部信息减少epistemic不确定性（模型知识不足），在长rollout中缓解复合误差。类似工作显示外生显著改善置信区间。[[13]](https://arxiv.org/abs/2301.13629)[[14]](https://github.com/yyysjz1997/awesome-timeseries-spatiotemporal-diffusion-model)

- **Title**: Scalable spatiotemporal prediction with Bayesian neural fields (BayesNF, 相关MVeLMA多模态UQ工作)
  **Authors**: Feras A. Saad 等（MVeLMA ~2025）
  **Year**: 2024
  **Venue**: Nature Communications
  **Contribution**: 结合神经场与分层贝叶斯推断，实现可扩展ST预测与良好校准的不确定性（MAP/VI ensembles）；外生协变量/傅里叶特征帮助减少epistemic不确定性（更好参数估计），同时改善aleatoric覆盖，适用于气候/公共健康多模态场景。[[15]](https://iopscience.iop.org/article/10.1088/1748-9326/ada397)

**7. Lead time分析——外生信息益处随预测horizon（1天 vs 1周 vs 1月 vs 1季）的变化**

上述多篇论文（尤其是ExoST 2025、ExoTST 2024、RiverMamba 2025、TimeXer 2024及Survey 2023）均包含不同horizon的消融实验，通常显示：**短horizon（1天）内历史自回归信号已足够，外生增益有限；中长horizon（1周至1月）外生信号（如天气、事件）显著纠正漂移、降低MSE；季节级长时预测中外生多尺度信息（趋势/宏观事件）益处最大，同时减少不确定性**。ExoST明确指出外生对长horizon漂移的纠正作用随lead time增加而增强；扩散/概率模型（如DiffSTG、GenCast类似工作）在长lead time下通过多模态外生改善集合预测覆盖度。[[7]](http://home.ustc.edu.cn/~zzy0929/Home/Paper/WWW25_ExoLLM.pdf)[[2]](https://arxiv.org/abs/2509.05779)

**补充资源**：Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth (Wei Chen, Yuxuan Liang等, NeurIPS 2024) 提供全球多模态（时间序列+geo-images+文本）数据集，支持零样本与长时ST预测研究。Awesome TS/STF论文列表（GitHub ddz16/TSFpaper及Qingsong Wen资源）包含更多2023-2026扩展（如ExoLLM使用LLM处理外生语言信号的长时预测）。[[16]](https://neurips.cc/virtual/2024/poster/97768)

这些论文主要来自arXiv、NeurIPS、IEEE、SIGSPATIAL等，覆盖了查询的所有子主题。建议直接访问arXiv PDF获取代码与详细实验（多数开源）。如果需要特定论文的更深入总结或PDF浏览结果，可进一步提供


**多模态时空预测中的泛化、解释性、基准和开放问题相关研究论文（2023-2026）**[[1]](https://neurips.cc/virtual/2024/poster/97768)[[2]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a6a7fbd1ee0c9684b3f919f79d129ef-Abstract-Datasets_and_Benchmarks_Track.html)

以下是针对查询主题（泛化、解释性、基准、开放挑战）筛选出的代表性论文（主要来自arXiv、NeurIPS、AAAI、ICML等，聚焦2023-2026年，与CityMind-Lab/Yuxuan Liang、Microsoft Research (ClimaX)、Aditya Grover、DeepMind、USTC (ExoLLM)等群体高度相关）。我优先选择了直接涉及零样本/少样本、域适应/OOD（例如COVID交通模式）、因果发现、SHAP适配、多模态基准（如Terra）、评估指标超越MSE/MAE、异构模态对齐、缺失模态、可扩展性以及人类-AI协作等子主题的论文。每篇包括标题、作者、年份、 venue、1-2句贡献总结及URL/arXiv ID。信息基于公开搜索和论文摘要/关键部分提取。[[3]](https://ojs.aaai.org/index.php/AAAI/article/view/38683/42645)[[4]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html)

### 1. Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth
- **作者**：Wei Chen, Xixuan Hao, Yuankai Wu, Yuxuan Liang (CityMind-Lab)。
- **年份/Venue**：2024, NeurIPS 2024 (Datasets and Benchmarks Track)。
- **贡献**：提出了大规模全球多模态时空数据集（覆盖648万网格、45年小时级气象时间序列 + 地理图像 + 解释性文本），解决了现有数据集（如METR-LA、PeMS、ERA5）在规模、多模态融合和跨区域泛化上的局限，支持基准测试、零样本/少样本研究及超越传统MSE/MAE的空间一致性、时间连贯性评估。极大促进多模态时空预测基准和开放挑战研究。[[1]](https://neurips.cc/virtual/2024/poster/97768)
- **URL/arXiv**：https://github.com/CityMind-Lab/NeurIPS24-Terra；NeurIPS proceedings / OpenReview: I0zpivK0A0。

### 2. Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment (CaST)
- **作者**：Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann。
- **年份/Venue**：2023, NeurIPS 2023。
- **贡献**：提出CaST框架，通过结构因果模型（SCM）、后门调整（处理时间OOD，如COVID-era交通模式变化）和前门调整+Hodge-Laplacian（动态空间因果），提升时空图预测的OOD泛化能力和解释性，提供因果强度热图等可视化解释，优于基线并直接支持因果发现和不变学习。[[4]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html)
- **URL/arXiv**：arXiv:2309.13378；https://github.com/yutong-xia/CaST。

### 3. ClimaX: A Foundation Model for Weather and Climate
- **作者**：Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K. Gupta, Aditya Grover (Microsoft Research/AdityaLab相关)。
- **年份/Venue**：2023, arXiv:2301.10343 (ICML 2023)。
- **贡献**：首个天气/气候基础模型，支持异构数据集（不同变量、时空分辨率、物理 grounding，如ERA5/CMIP6）的预训练，通过变量标记化和聚合实现跨任务、跨区域的强泛化（类似零/少样本微调），在WeatherBench、ClimateBench等基准上SOTA，处理多模态异构数据并支持域适应；为全球尺度模型的可扩展性提供基础（与DeepMind GraphCast互补，后者2023 Science聚焦中程天气预报）。[[5]](https://arxiv.org/abs/2301.10343)
- **URL/arXiv**：https://arxiv.org/abs/2301.10343；https://microsoft.github.io/ClimaX/。

### 4. Causal Spatio-Temporal Prediction: An Effective and Efficient Multi-Modal Approach (E²-CSTP / CSTP)
- **作者**：Yuting Huang, Ziquan Fang, Zhihao Zeng, Lu Chen, Yunjun Gao。
- **年份/Venue**：2025, NeurIPS 2025 (arXiv:2505.17637)。
- **贡献**：提出多模态（序列+文本/图像）框架，通过跨模态融合、双分支因果推理（DeepSHAP构建因果邻接矩阵、干预去除混杂）和高效ST编码，实现预测准确性提升与模型解释性（SHAP适配、因果结构可视化）；直接针对因果 vs 相关外生信号、OOD泛化和计算效率，优于单模态方法。[[6]](https://papers.nips.cc/paper_files/paper/2025/file/b71169b4db5448c494c91db848f060be-Paper-Conference.pdf)[[7]](https://arxiv.org/abs/2505.17637)
- **URL/arXiv**：https://arxiv.org/abs/2505.17637。

### 5. ST-VLM: A Spatial-to-Image Multimodal Spatial-Temporal Prediction Framework with Vision-Language Model
- **作者**：Tong Zhao 等。
- **年份/Venue**：2026, AAAI 2026。
- **贡献**：将视觉-语言模型用于时空预测，通过空间到图像的多模态表示编码动态空间依赖并融合多模态信息，在多个基准上取得SOTA，并在少样本场景中展现强大泛化能力；支持注意力可视化相关解释，是多模态LLM在ST预测中的早期工作，缓解预定义图结构限制和分布偏移问题。[[3]](https://ojs.aaai.org/index.php/AAAI/article/view/38683/42645)
- **URL/arXiv**：AAAI proceedings / ojs.aaai.org (相关arXiv变体存在)。

### 6. ExoLLM: Exploiting Language Power for Time Series Forecasting with Exogenous Variables (相关USTC工作)
- **作者**：Q. Huang 等 (USTC/ExoLLM相关)。
- **年份/Venue**：2025, WWW 2025 (或密切相关arXiv)。
- **贡献**：利用LLM处理外生变量，通过元指令、多粒度提示和双TS-文本注意力实现模态对齐，区分因果 vs 相关外生信号，提升多模态（数值+文本）预测的鲁棒性和泛化；直接针对开放挑战中的外生信号因果性与异构模态对齐。[[8]](https://ojs.aaai.org/index.php/AAAI/article/view/42314/46275)
- **URL/arXiv**：相关arXiv（如Augur等系列）或WWW proceedings。

### 7. What If TSF: A Benchmark for Reframing Forecasting as Scenario-Guided Multimodal Forecasting (WIT / WhatIfTSF)
- **作者**：J. Jang 等。
- **年份/Venue**：2026, arXiv:2601.08509 (相关会议)。
- **贡献**：提出“what-if”场景引导的多模态预测基准，包含专家反事实情景，用于评估外生事件缺失情况下的预测（“如果外生事件没发生会怎样？”）；推进反事实推理、OOD泛化、人类-AI协作解释性，并提供超越传统指标的评估，支持leaderboard式基准比较。[[9]](https://aaai.org/2026-ai-for-social-impact-presentations/)
- **URL/arXiv**：arXiv:2601.08509；相关GitHub。

### 附加参考（Survey/Benchmark补充）
- **Multi-modal Time Series Analysis: A Tutorial and Survey** (arXiv:2503.13709, 2025)：全面调研多模态TS，包括域泛化、常见基准（讨论Terra等）、挑战（如缺失模态、不规则采样、异步对齐）和未来方向（因果、人类-AI），为开放问题提供系统视图。[[10]](https://arxiv.org/html/2503.13709v1)

这些论文覆盖了关键群体的工作：Liang组在Terra/CaST上的因果+基准贡献、Microsoft ClimaX的泛化基础模型、USTC的ExoLLM、DeepMind GraphCast的延伸解释性工作等。许多论文在METR-LA/PeMS/ERA5/Terra上评估，并探索空间一致性、时间连贯性等指标；竞赛/leaderboard方面，WeatherBench、ClimateBench及Terra推动的平台是主流。

**研究差距矩阵（Research Gap Matrix）**
基于上述论文，最被低估/未充分探索的问题及建议的论文/提案方向如下（按探索程度分级：高/中/低）：

- **零样本/少样本跨区域预测 & 域适应**：中等探索（ClimaX异构预训练、ST-VLM少样本、Terra数据支持）。**最未探索子方向**：异步多模态（图像+文本+不规则TS）下的真正零样本全球迁移。**提案方向**：构建Terra-based geo-prompt基础模型或对比域适应框架，支持跨气候/城市零样本迁移（硕士/博士论文可聚焦Terra微调+不变表示）。

- **OOD泛化（COVID-era等） & 不变学习**：较高探索（CaST因果干预直接处理时间OOD和动态空间因果）。**差距**：多模态外生冲击下的联合不变学习仍不足。**提案**：开发多模态因果不变表征学习，结合SHAP验证，在真实OOD数据集（如疫情+气候事件）上测试。

- **解释性（注意力可视化、因果发现、反事实、SHAP/LIME适配）**：增长中（CaST因果透镜、E²-CSTP DeepSHAP因果图、WIT反事实基准、ExoLLM提示解释）。**最未探索**：跨模态（async文本/图像/TS）全球模型的可扩展注意力可视化和集成反事实模拟器；SHAP/LIME在高维时空多模态上的高效适配不足。**提案方向**： “多模态时空反事实模拟器与XAI人类-AI协作框架”（博士课题，结合WIT基准+Terra数据，实现“what if”科学假设测试）。

- **基准（Terra、METR-LA/PeMS/ERA5等；超越MSE/MAE的空间/时间一致性；竞赛/leaderboard）**：良好推进（Terra全球多模态基准、Time-IMM不规则基准、WIT反事实benchmark、WeatherBench/ClimaX leaderboard）。**差距**：缺少综合 leaderboard 覆盖OOD、缺失模态、因果指标和人类评估。**提案**：扩展Terra为持续开放平台，添加空间连贯性/因果分数指标和年度竞赛。

**最未探索的开放挑战（Open Challenges）及整体差距**：
- **异构模态对齐（异步来源）、缺失模态 & 不规则采样**：Time-IMM等开启，但全球尺度（Terra数据 vs 模型）仍严重不足（低探索）。
- **因果 vs 相关外生信号**：ExoLLM/E²-CSTP有进展，但高维多模态下的稳健区分仍开放。
- **全球尺度模型的计算可扩展性**：ClimaX/GraphCast在天气上较好，城市/多模态（图像+文本）全球模型的效率差距大。
- **人类-AI协作科学预测**：几乎未触及（最低探索），是最大空白。

**建议论文/提案方向**（优先高影响）：
1. **“因果多模态时空基础模型：整合LLM、SHAP因果发现与反事实推理”**（构建于E²-CSTP + ExoLLM + Terra + WIT，针对解释性和因果 vs 相关挑战；适合PhD，目标 NeurIPS/ICML）。
2. **“全球时空预测的OOD泛化与模态鲁棒性基准与增强”**（零样本跨区域、缺失模态处理，使用扩展Terra评估空间/时间一致性；含leaderboard贡献）。
3. **“科学预测中的人类-in-the-Loop可解释AI：反事实探索与协作优化”**（填补人类-AI最大差距，结合注意力可视化与反事实模拟，用于气候/城市系统）。
4. **不变学习与异步模态对齐框架的可扩展性研究**（针对全球模型，测试真实世界不规则采样和外生事件）。

总体而言，领域正向基础模型（ClimaX-style）+因果解释性+多模态基准（Terra驱动）演进，但异步异构对齐、可扩展人类-AI系统以及全面反事实工具仍是2026+的主要论文/论文机会。建议进一步阅读Terra GitHub和相关survey以获取最新代码/数据集。[[11]](https://github.com/AdityaLab/MM4TSA)

（以上基于公开web搜索和论文提取，如需特定PDF深入分析或更多论文，可进一步扩展。）


