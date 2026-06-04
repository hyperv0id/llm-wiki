---
title: "UniFlow"
type: technique
tags:
  - spatiotemporal
  - foundation-model
  - traffic-forecasting
  - transformer
  - retrieval-augmentation
  - arxiv-2024
created: 2026-05-31
last_updated: 2026-06-01
source_count: 1
confidence: medium
status: active
---

# UniFlow

**UniFlow** 是清华大学 FIB Lab（Yuan Yuan, Jingtao Ding 等）提出的首个统一 grid-based 和 graph-based 城市时空流预测的基础模型（foundation model），arXiv:2411.12972 (2024) [^src-uniflow]。与同实验室的 [[urbandit|UrbanDiT]]（NeurIPS 2025，扩散范式）和 UniST（KDD 2024，仅 grid）一脉相承，UniFlow 采用纯 Transformer + 可学习 memory retrieval 路线，在所有 9 个数据集上以单一模型超越专用基线 [^src-uniflow]。

## 核心设计

### 1. 多视角时空 Patching

将异构数据格式统一为 sequences [^src-uniflow]：

| 数据类型 | Patching 方法 | 输出 |
|----------|-------------|------|
| **Grid-based** (T×H×W) | 3D-CNN (kernel=pt×ps×ps, stride 同名) | L = T/pt × H/ps × W/ps 个 patches，扫描线 flatten |
| **Graph-based** (T×N) | 时间：1D-CNN；空间：METIS 图分割（最小化割边+平衡子图大小）→ 子图内 mean pooling | L 个子图 patches |

采用 **channel-independence** 策略：T×N×C → C 个独立 T×N，与 [[patchtst|PatchTST]] 和 [[channel-independence|CI 范式]] 一致 [^src-uniflow]。

### 2. 时空 Transformer

标准 encoder-decoder 架构 [^src-uniflow]：
- **Encoder**（4 layers, hidden=256）：仅处理历史数据 Sh = MASK(S)，mask 未来位置
- **Decoder**（4 layers）：接收 CONCAT(encoder output Ze, mask token embeddings Em)，全序列自注意 + MRA 增强 prompts

与 [[urbandit|UrbanDiT]] 不同，UniFlow 不依赖扩散过程——纯自回归预测路径，避免了 DDPM 的多步采样开销 [^src-uniflow]。

### 3. ST-MRA（Spatio-Temporal Memory Retrieval Augmentation）

UniFlow 最具创新性的组件——受 NLP 中 [[retrieval-augmented-generation|RAG]] 启发，但完全可学习（非参数检索），使不同数据类型的共享时空模式能跨学习 [^src-uniflow]：

#### 四组 Memory

| Memory | 内容 | Query 来源 |
|--------|------|-----------|
| Time-domain (T) | 时序动态模式 | Self-Attention(Sh) |
| Frequency-domain (F) | 周期/循环行为 | FFT(Sh) → 幅度+相位 |
| Time-spatial (ST) | 基于时域自适应的空间关系 | GCN(Et, At=softmax(ReLU(Et·Etᵀ))) |
| Frequency-spatial (SF) | 基于频域自适应的空间关系 | GCN(Ef, Af=softmax(ReLU(Ef·Efᵀ))) |

每 memory 为 N × D learnable key-value embeddings（N=512, D=256）[^src-uniflow]。

#### 检索与增强流程

```
Query formulation (temporal + spatial queries from input)
  → Cosine similarity: α = softmax(Q · Kᵀ)
  → Weighted aggregation: P = Σ αⱼ Vⱼ
  → Augmentation: decoder input += P
```

四组 prompts {Pt, Pf, Pst, Psf} 直接加到 decoder input embeddings，每个 prompt 提供不同视角的时空先验 [^src-uniflow]。

## 性能

### 短时预测（12→12）

| 数据集 | UniFlow RMSE | Best Baseline | 提升 |
|--------|-------------|---------------|------|
| TaxiBJ (grid) | 20.35 | 23.67 (UniST) | ↓14.0% |
| FlowSH (grid) | 15.18 | 19.95 (UniST) | ↓23.9% |
| CrowdNJ (grid) | 0.180 | 0.191 (UniST) | ↓5.8% |
| TaxiNYC (grid) | 16.02 | 17.55 (UniST) | ↓8.7% |
| TrafficBJ (graph) | 1.72 | 1.76 (MTGNN) | ↓2.3% |
| TrafficSH (graph) | 1.84 | 1.85 (MTGNN) | ↓0.5% |

平均相对提升 **9.1%** RMSE vs 最佳 baseline [^src-uniflow]。

### 长时预测（64→64）

相对最佳 baseline 平均提升 **11.9%** RMSE。视频预测模型（MAU/MIM/SimVP）长时上有所改善但仍不敌 UniFlow [^src-uniflow]。

### 小样本/零样本

- **10% few-shot**：接近 full-data 性能
- **5% few-shot**：显著优于所有 baseline 的 5%/10% 表现
- **零样本**：超越多数有训练数据的 baseline [^src-uniflow]

### 鲁棒性

1%/5%/10% 高斯噪声扰动下 UniFlow 性能几乎不变（TaxiBJ 10% 噪声 RMSE=22.97 vs clean=20.83），而 [[patchtst|PatchTST]] 从 55.37 崩溃到 137.5（↑148%）——验证了大规模跨数据集预训练带来的鲁棒性红利 [^src-uniflow]。

## 与其他时空基础模型的对比

| 特征 | **UniFlow** | [[urbandit|UrbanDiT]] | UniST | [[urbangpt|UrbanGPT]] | [[opencity|OpenCity]] |
|------|------------|-----------|-------|----------|----------|
| 数据类型 | Grid + Graph | Grid + Graph | Grid only | Grid only | Grid only |
| 范式 | Transformer + MRA | Diffusion Transformer | Transformer + prompt | LLM + 指令微调 | Transformer + 实例归一化 |
| 零样本 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 大规模 Graph (>10K nodes) | ✓ | — | — | — | — |
| 推理方式 | 自回归 | 扩散去噪 | 自回归 | LLM 一次生成 | 非自回归 |
| 参数规模 | ~几M (hidden=256) | S/M/L (up to 12 layers) | — | 7B | 2M/5M/26M |
| 发布时间 | arXiv Nov 2024 | NeurIPS 2025 | KDD 2024 | KDD 2024 | arXiv Aug 2024 |

UniFlow 在同实验室的产品线中填补了"**纯 Transformer + 大规模 Graph 支持**"的空白——UrbanDiT 是扩散路线，UniST 是 grid-only [^src-uniflow]。

## Connection

UniFlow 与以下页面紧密关联：

- [[spatio-temporal-foundation-model]] — 时空基础模型概念，UniFlow 是其中 grid+graph 统一路径的代表
- [[traffic-forecasting]] — 交通预测，UniFlow 的三大 graph 数据集（>10K nodes）扩展了该领域的规模基准
- [[urbandit]] — UrbanDiT，同实验室的扩散范式时空基础模型（NeurIPS 2025）
- [[urbanfm]] — UrbanFM，scaling 为核心的系统性时空基础模型框架（WorldST+MiniST+极简 Transformer，arXiv 2026）
- [[factost]] — FactoST，因子化时空基础模型（UTP+STA），同 HKUST(GZ) group
- [[urbangpt]] — UrbanGPT，LLM-based 时空预测（KDD 2024）
- [[opencity]] — OpenCity，zero-shot 交通预测基础模型
- [[gpt-st]] — GPT-ST，时空图 MAE 预训练（NeurIPS 2023）
- [[stgcn]] — STGCN，被 UniFlow 用作 graph baseline
- [[dcrnn]] — DCRNN，被 UniFlow 用作 graph baseline
- [[gwnet]] — GWNet，被 UniFlow 用作 graph baseline
- [[patchtst]] — PatchTST，被 UniFlow 用作多变量时序 baseline + channel-independence 来源
- [[itransformer]] — iTransformer，被 UniFlow 用作多变量时序 baseline
- [[channel-independence]] — CI 策略，UniFlow 在两阶段 patching 中采用
- [[ustd]] — USTD，任务统一型 ST diffusion 框架（SIGSPATIAL 2024），与 UniFlow 的 foundation model 路线互补——USTD 的 GSM 预训练策略可用于 foundation model 编码器训练
- [[bigcity]] — BIGCity，首个 MTMD 时空模型，将 UniFlow 的 traffic-only 范式扩展到轨迹+交通状态联合处理（arXiv 2024）
- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：先验注入派/纯 Transformer 路线代表

## 局限性

- 仅 9 数据集验证，覆盖城市/任务类型有限 [^src-uniflow]
- ST-MRA memory 数量和类型手动设计，未来可探索自动化配置 [^src-uniflow]
- 纯数值，未融合多模态（文本/图像），与 [[most|MoST]]、[[aurora|Aurora]] 等形成互补路线 [^src-uniflow]
- 短时预测在 TrafficSH graph 上提升有限（0.5%），graph 数据的边际收益小于 grid [^src-uniflow]

[^src-uniflow]: [[source-uniflow]]
