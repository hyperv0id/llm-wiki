# 多模态外生信息引导的长期时空预测：领域综述

## 一、问题定义与核心形式化

该方向旨在利用来自时空系统外部的多模态信息（气候状态、环境事件、文本报告、图表数据、社会活动等），辅助长期时空系统的未来演化预测与智能推理。

**形式化定义**：预测未来时空状态 \( Y_{t+1:T} = f(Y_{1:t}, X_{1:t}, E_{1:T}) \)，其中：
- \( Y \)：目标时空序列（如交通流量网格、气温场）
- \( X \)：内部协变量（如时间戳、节点属性）
- \( E \)：多模态外生信息（文本嵌入、图像特征、事件指标、气候指数等）

**与传统时空预测的区别**：传统方法（STGCN、GraphWaveNet 等）主要依赖历史自回归和空间图依赖。本方向显式整合外部驱动因素，强调跨模态长程因果/影响建模，以处理长期预测中的误差累积和外部冲击。

---

## 二、时空基础模型与多模态融合架构

### 2.1 时空基础模型对多模态外生的支持

| 模型 | 年份 | 外生支持方式 | 关键特点 |
|------|------|-------------|---------|
| **Aurora** | 2025 | 文本（BERT）+ 图像（ViT）+ 领域知识 | Token distillation + Modality-Guided Self-Attention；零样本跨域生成 |
| **ClimaX** | 2023 | 多气候变量 + 静态外生（地形等） | Variable tokenization；异构数据统一处理 |
| **UniST** / **UrbanDiT** | 2024 | 文本 POI、可学习 prompt 内存池 | Prompt learning；零样本多任务城市预测 |
| **Pangu-Weather** / **GraphCast** | 2023 | 多气象变量 + 静态外生 | 额外输入通道/掩码注入 Transformer/GNN |
| **TimeGPT** / **Moirai** / **TimesFM** / **Chronos** | 2024 | 数值协变量（dynamic covariates） | 从单变量向协变量/多模态条件化演进 |
| **Terra 数据集** | NeurIPS 2024 | 648 万全球网格 × 45 年气象 TS + 地理图像 + LLM 生成文本 | 为多模态 ST 基础模型提供核心基准 |

### 2.2 多模态融合架构

- **Cross-attention**：时序 + 文本 + 图像间的跨模态交互（E²-CSTP、Aurora）
- **Early fusion**：输入级特征拼接
- **Intermediate fusion**：潜在空间跨模态交互（Transformer 最常见）
- **Late fusion**：决策级融合
- **Mixture-of-Experts (MoE)**：模态路由，如 ExoST 的 latent space gated expert 模块
- **Gated fusion**：门控机制动态平衡不同模态/历史 vs 未来影响

### 2.3 外生信息编码与注入方式

- **文本**：BERT/LLM 嵌入 → cross-attention / addition / prompt 注入 ST backbone
- **图像**：ViT/CNN 提取 patch/token → token distillation 或 modality-guided attention
- **知识图谱**：GNN 嵌入实体/关系 → 节点特征或额外图结构与 ST-GNN 融合
- **物理方程**：纳入损失函数（PINN 风格）或作为硬约束融入神经算子（FNO/DeepONet）

**代表性框架**：**ExoST**（2025）提出 "select then balance" 范式——latent space gated expert 动态选择显著外生信号，siamese dual-branch + context-aware weighting 实现历史/未来平衡，即插即用兼容现有 ST backbone。

### 2.4 PINNs 与神经算子的多模态扩展

- **PI-MFM**（arXiv:2512.23056, 2025）：将物理方程直接纳入预训练和适配，支持多模态输入求解 PDE
- **Multimodal PINN**（arXiv:2503.08482, 2025）：融合数值特征 + 鱼眼图像 + 辐射传输方程到损失函数

---

## 三、LLM 驱动的多模态时空推理

### 3.1 LLM 作为推理引擎

**ExoLLM**（WWW 2025）是该方向的标杆工作：通过 Meta-task Instruction 激活 LLM 的语言知识，Multi-grained Prompts 捕捉新闻/事件的多样化动态影响，Dual TS-Text Attention 对齐文本-数值特征空间，避免仅依赖时序导致的虚假相关。

**GPT4MTS**（AAAI 2024）将数值时序和文本外生信息转化为结构化 prompt 输入 GPT-like LLM，实现多模态联合预测。

### 3.2 多模态 LLM/VLM 应用于时空数据

| 模型 | 模态 | 特点 |
|------|------|------|
| **ST-Vision-LLM** | 交通图像 + 时序 + 文本 | ViT 编码器 + 数值 tokenization + 两阶段微调（SFT+RL），长时预测提升 15.6% |
| **Solar-VLM** | 卫星图像 + 天气文本 + 时序 | 模态特定编码器 + GAT + 跨站点注意力 |
| **UrbanGPT** | 城市时空 + 文本 | ST 依赖编码器 + 指令微调，零样本多任务 |

### 3.3 Reasoning-Augmented 预测

- **From News to Forecast**（NeurIPS 2024）：LLM 生成式 Agent 迭代过滤新闻 + Reflection 机制评估影响 + 融合数值预测
- **Time-LLM**（ICLR 2024）：通过 reprogramming 将 TS patch 映射为文本原型，Prompt-as-Prefix 注入领域知识

### 3.4 RAG 与 Agent 方法

- **RAF**（arXiv:2411.08249, 2024）：原则性 RAG 框架，检索历史模式/外生知识增强 LLM 零样本预测
- Agent 方法：LLM 代理查询数据库、运行模拟、多轮 reflection 合成时空预测

---

## 四、长期时空依赖建模与不确定性量化

### 4.1 误差累积与漂移纠正

**ExoST**（2025）首次系统研究外生变量建模的两个核心挑战：不同变量影响不一致、历史与未来数据不平衡。通过 select-then-balance 范式有效缓解自回归长时漂移。

**ExoTST**（arXiv:2410.12184, 2024）将过去与未来外生变量视为不同模态，使用跨时间融合模块整合到自回归预测，对比 PatchTST、iTransformer 等显著减少长 horizon 误差累积。

### 4.2 长序列架构

- **TimeXer**（NeurIPS 2024）：patch-wise 自注意力（内生）+ variate-wise 跨注意力（外生），在周级长序列上优于 Informer/Autoformer/FEDformer 家族
- **Informer / Autoformer / FEDformer / PatchTST / TimesNet**：通过分解、频率增强、patching、多周期等方式处理长序列，外生信号通过特征拼接或独立编码器注入

### 4.3 状态空间模型（Mamba/S4/S5）

**STG-Mamba**（arXiv:2403.12418, 2024）首次将选择性 SSM 用于时空图预测，线性复杂度下捕捉长程依赖，可通过节点特征或外部输入实现多模态条件化。**Multi-scale Mamba**（2026）进一步实现层次化多尺度建模，支持不同尺度外生影响。

### 4.4 Neural ODE / 连续时间模型

**RainODE**（2026）：将降水预测重构为连续时间动力系统，在潜在空间中使用 Neural ODE，适合不规则采样时空数据和外生事件流的连续演化驱动。

### 4.5 不确定性量化

| 方法 | 代表工作 | 特点 |
|------|---------|------|
| **扩散模型** | DiffSTG (SIGSPATIAL 2023) | 去噪扩散 + STGNN，良好校准的概率预测；外生条件化减少 aleatoric 不确定性 |
| **贝叶斯方法** | BayesNF (Nature Comms 2024) | 神经场 + 分层贝叶斯推断；外生协变量减少 epistemic 不确定性 |
| **Deep Ensembles** | DYffusion (NeurIPS 2023) | 扩散模型用于动态系统，多模态外生改善集合预测覆盖度 |

### 4.6 Lead Time 分析

**一致结论**（ExoST、ExoTST、RiverMamba、TimeXer 等消融实验）：
- **短 horizon（1 天内）**：历史自回归信号足够，外生增益有限
- **中长 horizon（1 周–1 月）**：外生信号（天气、事件）显著纠正漂移，降低 MSE
- **季节级长时预测**：外生多尺度信息（趋势/宏观事件）益处最大，同时大幅减少不确定性

---

## 五、泛化能力、可解释性与开放挑战

### 5.1 泛化能力

- **CaST**（NeurIPS 2023）：通过结构因果模型 + 后门/前门调整，处理时间 OOD（COVID 交通模式），提升 ST 图预测的 OOD 泛化
- **ClimaX**（2023）：通过变量标记化和聚合实现跨任务、跨区域泛化（即零/少样本微调）
- **ST-VLM**（AAAI 2026）：VLM + 空间到图像表示，在少样本场景展现强大泛化

### 5.2 可解释性

- **CaST**（NeurIPS 2023）：因果强度热图可视化，直接支持因果发现
- **E²-CSTP**（2025）：DeepSHAP 构建因果邻接矩阵 + 干预去除混杂
- **WIT/WhatIfTSF**（arXiv:2601.08509, 2026）："What if" 场景引导的反事实基准，评估外生事件缺失下的预测

### 5.3 关键基准数据集

| 数据集 | 模态 | 规模 | 来源 |
|--------|------|------|------|
| **Terra** | 气象 TS + 地理图像 + 解释文本 | 648 万网格 × 45 年 | NeurIPS 2024 |
| **METR-LA / PeMS** | 交通 + 天气/事件外生 | — | 交通领域标准 |
| **ERA5** | 气候再分析 + 卫星 | 全球 | 气候领域标准 |
| **WIT/WhatIfTSF** | 反事实情景 | — | arXiv 2026 |

### 5.4 关键研究团体

| 团体 | 代表工作 | 重点方向 |
|------|---------|---------|
| **CityMind-Lab / Yuxuan Liang** | Terra, ExoST, CaST | 多模态 ST 基准 + 因果 + 外生建模 |
| **Tsinghua FIB Lab / Yong Li** | UniST, UrbanDiT | Prompt-based 通用城市 ST 模型 |
| **Microsoft Research / Aditya Grover** | ClimaX | 天气/气候基础模型 |
| **DeepMind** | GraphCast | 中程天气预报 |
| **USTC** | ExoLLM | LLM + 外生变量 ST 预测 |
| **AdityaLab** | MM4TSA 论文列表 | 多模态 TS 综述与资源汇总 |

---

## 六、关键论文汇总

| 论文 | 年份 | 会议/期刊 | 核心贡献 |
|------|------|----------|---------|
| ExoST | 2025 | arXiv | Select-then-Balance 外生变量即插即用框架 (arXiv only, 未经同行评审) |
| E²-CSTP | 2025 | NeurIPS | 因果多模态 + DeepSHAP + GCN+Mamba |
| Terra | 2024 | NeurIPS | 全球多模态 ST 基准数据集 |
| Aurora | 2025 | ICLR sub. | 首个多模态 TS 基础模型 |
| UrbanDiT | 2024 | arXiv | 扩散 Transformer 开放世界城市 ST |
| ExoLLM | 2025 | WWW | LLM 捕捉外生变量动态影响 |
| GPT4MTS | 2024 | AAAI | Prompt-based LLM 多模态 TS |
| ST-Vision-LLM | 2025 | arXiv | Vision-LLM 时空交通预测 |
| UniST | 2024 | KDD | Prompt-empowered 通用城市 ST |
| CaST | 2023 | NeurIPS | 因果透镜 + OOD 泛化 ST 图预测 |
| ClimaX | 2023 | ICML | 天气/气候基础模型 |
| STG-Mamba | 2024 | arXiv | Mamba 时空图长程依赖 |
| DiffSTG | 2023 | SIGSPATIAL | 扩散模型概率 ST 预测 |
| BayesNF | 2024 | Nature Comms | 贝叶斯神经场可扩展 ST 预测 |
| TimeXer | 2024 | NeurIPS | Patch-wise + variate-wise 外生注入 |

---

## 七、研究空白与选题建议

### 7.1 研究差距矩阵

| 挑战 | 探索程度 | 关键空白 |
|------|---------|---------|
| 短期交通/空气质量预测 | 高 | 方法较成熟 |
| 长期（>1 周）跨模态机制 | 低 | **严重不足**，证据薄弱 |
| 异构模态异步对齐 | 低 | 全球尺度（Terra vs 模型）仍严重不足 |
| 因果 vs 相关外生信号区分 | 中 | 高维多模态下稳健区分仍开放 |
| 零样本/少样本跨区域泛化 | 中 | 异步多模态下的真正零样本全球迁移不足 |
| 全球尺度计算可扩展性 | 中 | 城市/多模态全球模型效率差距大 |
| 人类-AI 协作科学预测 | **极低** | **最大空白** |

### 7.2 建议选题方向

1. **因果多模态时空基础模型**：整合 ExoST 的 select-then-balance + E²-CSTP 的因果推理 + LLM 外生理解，在 Terra 基准上开发，目标 NeurIPS/ICML

2. **面向异步异构模态的零样本全球 ST 预测**：基于 Terra 构建 geo-prompt 基础模型或对比域适应框架，支持跨气候/城市零样本迁移

3. **科学预测中的人机协同可解释 AI**：结合 CaST 因果可视化 + WIT 反事实基准 + LLM 自然语言交互，填补人机协同最大空白

4. **多模态外生驱动的长期不确定性量化**：DiffSTG/BayesNF 风格 + 多尺度外生条件化，系统研究不同 lead time 下的不确定性减少机制

5. **Mamba + 多模态融合的轻量长程 ST 预测**：STG-Mamba 路线 + ExoST 外生注入，面向计算效率与长 horizon 的实用方案

---

## 参考资源

- MM4TSA 论文列表：https://github.com/AdityaLab/MM4TSA
- Awesome ST Foundation Models：https://github.com/LMissher/Awesome-Spatio-Temporal-Foundation-Models
- Terra 数据集：https://github.com/CityMind-Lab/NeurIPS24-Terra
- Awesome TS/STF Papers：https://github.com/ddz16/TSFpaper

---

*报告日期：2026-07-08 | 方法：4 路并行网络检索 + 交叉验证 | 覆盖时间：2023-2026*
