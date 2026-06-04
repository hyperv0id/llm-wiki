---
title: "UrbanPG"
type: technique
tags:
  - spatial-temporal
  - prompt-learning
  - linear-attention
  - large-scale
  - few-shot
  - continual-learning
  - foundation-model
  - traffic-forecasting
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# UrbanPG

**UrbanPG** 是一个高效的、可扩展的城市时空学习框架（AAAI 2026），由同济大学的 Aoyu Liu 和 Yaying Zhang 提出[^src-urbanpg]。它通过将"个性化上下文提示"（[[personalized-context-prompt]]）和"通用时空骨干"拆分为两个独立且交互的组件，同时解决了城市时空预测中的三个核心挑战：大规模预测（8600 节点）、小样本泛化（仅 10% 训练数据）、持续学习（节点增量扩展而不遗忘）[^src-urbanpg]。

## 核心设计理念：解耦个性化与通用

UrbanPG 的核心洞察是：传统 [[traffic-forecasting|STGNN]] 的所有参数都绑定到特定场景的时空上下文中（节点数、采样周期、区域模式），导致"强场景依赖"[^src-urbanpg]。UrbanPG 的解法是将"场景特有模式"和"跨场景通用模式"拆成两块：

- **通用骨干**：轻量、场景无关，参数不依赖当前时空上下文，可冻结后在所有下游场景中复用。核心是 STCA（Spatio-Temporal Context Attention）线性注意力模块[^src-urbanpg]。
- **个性化上下文提示**：可学习的时间 embedding（Etod+Edow）和空间 embedding（Es），捕获当前场景的特有异质性。通过随机扰动正则化防止大 N 过拟合[^src-urbanpg]。

这种解耦直接借鉴了 [[stid|STID]]（CIKM 2022）的核心发现——在时空预测中"区分不同实体的上下文"比"精细建模实体间关系"的边际收益更大——但 UrbanPG 通过加入通用骨干补上了 STID 缺失的空间依赖建模[^src-urbanpg]。

## 三大组件

### 1. STCA 线性时空上下文注意力

STCA 将自注意力复杂度从 O(N²) 降为 O(N·d²)，在 CA（8600 节点）上实现了实际的线性行为[^src-urbanpg]：

$$H_{st} = \phi(Q) \cdot (\phi(K)^T V + \phi(P_t)^T V + \phi(P_s)^T V)$$

三项分别对应自注意力（Ah）、时间交叉注意力（At）、空间交叉注意力（As），均在 d×d 空间计算。φ 是 Performers 的随机特征映射，将 softmax 核函数近似为正交随机投影 + sin/cos 编码[^src-urbanpg]。这在概念上属于 [[linear-attention-unified-framework|线性注意力]]的 Performers 分支，与 Mamba 的 SSM 形式不同——Mamba 通过遗忘门实现循环线性注意力，STCA 通过随机特征映射实现矩阵乘法顺序变换[^src-urbanpg]。

### 2. 个性化上下文提示

- **时间提示 Pt**：由 Etod（time-of-day）和 Edow（day-of-week）两个可学习 embedding 相加构造，通过时间位置的索引查表获得。t-SNE 可视化显示 Etod 呈清晰的环形周期结构[^src-urbanpg]。
- **空间提示 Ps**：由可学习 Es ∈ R^(N×d) 经随机扰动正则化构造。训练时以概率 p=0.1 随机替换节点身份——部分用共享 embedding Ed 替换，部分用随机噪声 Md 替换——迫使模型不能偷懒只靠 Es 记忆数值，必须依赖通用骨干提取空间模式。推理时 p=0，完整 Es 恢复[^src-urbanpg]。

### 3. 提示调整门控

Hpst = (Hst · (1 + Pt) + Ps) · Pt —— 三重门控（时间缩放、空间偏移、时间再调制），使得提示不仅是注意力查询的参与者，还直接控制特征的表达能力[^src-urbanpg]。

## 多范式能力

### 大规模预测

UrbanPG 在 LargeST 四子集（SD 716 节点、GBA 2352、GLA 3834、CA 8600）全面 SOTA。在 CA 上，GWNet 和 STWave 因 O(N²) 内存爆炸无法运行，UrbanPG 的 O(N·d²) 复杂度随节点数近似线性增长。训练时间比 PatchSTG 低 48.96%，推理时间低 72.44%，内存低 45.72%[^src-urbanpg]。

### 小样本泛化

冻结预训练骨干，仅重建和训练下游的个性化提示 Pt 和 Ps。在 CA-D3（480 节点，仅 10% 训练数据）上 MAE=18.28，超越 FlashST（18.91）和 STD-MAE（20.09）。[[stid|STID]] 在小样本任务上完全崩溃（MAE=22.90），因为它只有场景特有参数，无通用骨干可迁移[^src-urbanpg]。

### 持续学习

冻结骨干 M，每次增量阶段仅扩展 Ps 的新行参数（为新增节点添加），旧节点参数不动。在 PEMS-Stream（7 增量期，655→871 节点）上平均 MAE=10.77，超越了需要知识蒸馏的 EAC（13.49，20.2%提升）[^src-urbanpg]。

## 与相关工作的关系

| 维度 | UrbanPG | [[flashst|FlashST]] | [[urbangpt|UrbanGPT]] | [[urbanfm|UrbanFM]] | [[urbandit|UrbanDiT]] |
|------|---------|------|----------|----------|-----------|
| **核心理念** | 提示-骨干解耦 | Prompt tuning 单范式 | LLM 指令微调 | Scaling 为中心 | 扩散 Transformer |
| **空间建模** | 线性自注意力 O(N·d²) | 提示调优 | LLM 语义理解（无图） | 因子分解注意力 | DiT + 三 memory pool |
| **复杂度** | O(N·d²) → 线性 | 低 | O(N) 但 7B 参数 | O(N·T) 因子分解 | O(N²) → 25× 加速 |
| **多任务** | ❌ 不支持 | ❌ 任务独立 | 固定 3 任务 | ✓ 统一预测+填补 | ✓ 5 任务 |
| **持续学习** | ✓ 提示扩展，零遗忘 | ❌ | ❌ | ❌ | ❌ |
| **零样本** | ✓ 提示微调即可 | ✓ 提示调优 | ✓ LLM 推理 | ✓ 无微调 | ✓ |
| **推理效率** | 高（比 PatchSTG 快 72%） | 中 | 极慢（174s/传感器） | 高（Chronos 的 4× 快） | 高（20 步） |

UrbanPG 区别于 [[flashst|FlashST]] 的关键点是：FlashST 也做 prompt tuning，但骨干和提示的耦合度更高，迁移需要更复杂的分布映射机制[^src-urbanpg]。UrbanPG 的提示-骨干解耦更彻底，且额外支持持续学习范式。UrbanPG 最大的局限性是无法像 [[urbanfm|UrbanFM]] 那样支持多任务并行训练——本文明确自曝这一点是走向时空基础模型的核心障碍[^src-urbanpg]。

## Related Pages

- [[source-urbanpg]] — source summary page
- [[spatio-temporal-foundation-model]] — broader ST foundation model paradigm
- [[traffic-forecasting]] — traffic prediction task overview
- [[urbanfm]] — UrbanFM, scaling-centric ST foundation model with multi-task capability (arXiv 2026)
- [[urbangpt]] — UrbanGPT, first ST LLM with instruction-tuning (KDD 2024)
- [[urbandit]] — UrbanDiT, diffusion transformer for open-world ST prediction (NeurIPS 2025)
- [[opencity]] — OpenCity, open-source ST foundation model for traffic (arXiv 2024)
- [[bigcity]] — BIGCity, first MTMD ST model unifying trajectory + traffic (arXiv 2024)
- [[uniflow]] — UniFlow, unified grid+graph ST foundation model with ST-MRA (arXiv 2024)
- [[unist]] — UniST, first one-for-all grid-based ST foundation model (KDD 2024)
- [[linear-attention-unified-framework]] — Mamba ↔ Linear Attention unified framework
- [[large-scale-spatial-temporal-graph]] — large-scale ST graph challenges
- [[std-mae]] — STD-MAE, spatial-temporal-decoupled masked pre-training (IJCAI 2024)
- [[gwnet]] — GWNet, traditional O(N²) STGNN baseline compared by UrbanPG
- [[stid]] — STID, lightweight embedding-only baseline that inspired UrbanPG's context prompts
- [[urbanverse]] — UrbanVerse, foundation model for cross-city/cross-task region attribute prediction (crime/population/carbon), complementary urban ML paradigm (arXiv 2026)
- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：prompt-backbone 解耦路线

[^src-urbanpg]: [[source-urbanpg]]
