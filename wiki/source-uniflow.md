---
title: "Source: UniFlow"
type: source-summary
tags:
  - spatio-temporal
  - foundation-model
  - traffic-forecasting
  - transformer
  - rag
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# Source: UniFlow — A Foundation Model for Unified Urban Spatio-Temporal Flow Prediction

**Yuan Yuan, Jingtao Ding, Chonghua Han, Zhi Sheng, Depeng Jin, Yong Li** (Tsinghua University, BNRist, Department of Electronic Engineering). arXiv:2411.12972v3 (Nov 2024, revised Apr 2025). Code: <https://github.com/YuanYuan98/UniFlow>.

## 核心贡献

UniFlow 是首个统一 grid-based 和 graph-based 城市时空流预测的基础模型（foundation model）。传统方法为两类数据分别训练专用模型（CNN-based 用于 grid，GNN-based 用于 graph），UniFlow 通过单一 all-in-one 模型在两类数据上均取得 SOTA [^src-uniflow]。

三大组件：

1. **多视角时空 patching**：grid 数据通过 3D-CNN (kernel=pt×ps×ps) patching → 扫描线序列化；graph 数据通过 1D-CNN（时间）+ METIS 图分割（空间，最小化割边）→ 子图内 mean pooling → 序列化。采用 channel-independence 策略，T×N×C → C 个独立 T×N 序列 [^src-uniflow]。

2. **时空 Transformer**：标准 encoder-decoder 架构。Encoder 仅处理历史数据（mask 未来），decoder 接收 encoder 输出 + mask token embeddings 做全序列自注意。配置：4 encoder + 4 decoder layers，hidden=256 [^src-uniflow]。

3. **ST-MRA（Spatio-Temporal Memory Retrieval Augmentation）**：核心创新——受 RAG 启发但完全可学习的检索增强机制。维护 4 组结构化 memory（key-value pairs, N×D learnable embeddings）：time-domain memory（时序模式）、frequency-domain memory（周期模式，FFT 提取）、time-spatial memory（基于时域模式自适应的空间关系）、frequency-spatial memory（基于频域模式自适应的空间关系）。Query 公式化：时序 query = self-attention(Sh) + FFT(Sh)；空间 query 通过自适应学习图拓扑 At = softmax(ReLU(Et·Etᵀ)) 后 GCN 编码。检索：cosine similarity 匹配 query ↔ key → 加权聚合 value → 得到 prompts Pi，直接 add 到 decoder input embeddings [^src-uniflow]。

## 实验

**9 个数据集**（6 grid + 3 graph）：TaxiBJ、TaxiNYC、CrowdNJ、CrowdBJ、FlowSH、PopSH（grid）；TrafficBJ、TrafficSH、TrafficNJ（graph，>10,000 nodes 大规模）。对比 4 类 baseline：经典方法（HA/ARIMA）、grid 深度模型（STResNet/ACFM/STNorm/MAU/MIM/SimVP/TAU/PromptST/UniST）、graph 深度模型（STGCN/DCRNN/GWN/MTGNN/AGCRN/GTS/STEP）、多变量时序模型（PatchTST/iTransformer/Time-LLM）+ PatchTST-v2 (one-for-all) [^src-uniflow]。

**短时预测**（12→12）：UniFlow 在所有 6 个数据集上取得最优，相对最佳 baseline 平均提升 9.1% RMSE。grid 最佳 baseline=UniST (TaxiBJ RMSE=23.67)，UniFlow=20.35；graph 最佳 baseline=MTGNN (TrafficBJ RMSE=1.76)，UniFlow=1.72。PatchTST-v2（one-for-all）比单独 PatchTST 更差，说明简单统一训练无法自动学习跨数据集的互惠 [^src-uniflow]。

**长时预测**（64→64）：UniFlow 继续最优，相对最佳 baseline 平均提升 11.9% RMSE。视频预测模型（MAU/MIM/SimVP）在长时上有所改善但仍低于 UniFlow [^src-uniflow]。

**小样本/零样本**：CrowdSH 为目标数据集，其余为源数据集。10%/5% few-shot 微调下 UniFlow 接近 full-data 性能，baseline 显著退化；零样本性能超越几乎所有有训练数据的 baseline [^src-uniflow]。

**鲁棒性**：1%/5%/10% 高斯噪声扰动下，UniFlow 几乎稳如泰山（TaxiBJ 1% RMSE=20.83 → 10% RMSE=22.97），baseline 严重退化（PatchTST 1%=55.37 → 10%=137.5），验证了基础模型的大规模预训练鲁棒性优势 [^src-uniflow]。

**消融**：memory units 512 最优（太少欠表达，太多难聚类）；去除任一 memory 类型（T/F/ST/SF）均显著降性能；去除全部 MRA（w/o MRA）降幅最大 [^src-uniflow]。

**Case Study**：北京/上海的城市中心区域 memory retrieval 权重高度相似，偏远住宅区也高度相似——ST-MRA 学到的 retrieval 模式具有语义可解释性 [^src-uniflow]。

## 与相关工作的对比

| 模型 | 数据类型 | 零样本 | 模型类型 |
|------|---------|--------|---------|
| UniFlow | Grid + Graph 多数据集 | ✓ | 统一 Transformer |
| UniST | Grid 多数据集 | ✓ | 统一 Transformer |
| UrbanGPT | Grid 多数据集 | ✓ | LLM-based |
| UrbanDiT | Grid + Graph 多数据集 | ✓ | Diffusion Transformer |
| GPT-ST | Graph 单数据集 | ✗ | MAE 预训练 |

UniFlow 的独特性：**唯一同时支持 grid+graph 多数据集统一训练且 zero-shot 的纯 Transformer 基础模型**（UrbanDiT 基于扩散范式，UniST 仅 grid）[^src-uniflow]。

## 局限性

- 仅在 9 个数据集上验证，覆盖城市和任务类型有限 [^src-uniflow]
- ST-MRA 的 memory 数量和类型为手动设计，缺乏自动化配置 [^src-uniflow]
- 未探索多模态数据融合（文本/图像等）[^src-uniflow]
- 未来方向：扩展到气象学/地球科学等更复杂的时空场景 [^src-uniflow]

[^src-uniflow]: [[source-uniflow]]
