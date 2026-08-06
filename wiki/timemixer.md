---
title: "TimeMixer"
type: entity
tags:
  - time-series
  - forecasting
  - multiscale
  - mlp
  - iclr-2024
created: 2026-05-31
last_updated: 2026-08-06
source_count: 3
confidence: medium
status: active
---

# TimeMixer

**TimeMixer** 是蚂蚁集团和清华大学联合提出的全 MLP 多尺度时序预测模型（ICLR 2024）。它从多尺度混合的新视角处理时间序列中复杂的时序变异，通过 Past-Decomposable-Mixing（PDM）历史信息提取模块和 Future-Multipredictor-Mixing（FMM）未来预测模块，在长期和短期预测任务上均取得一致的最优性能[^src-timemixer]。

## 核心设计

### 多尺度序列生成

输入序列 $x \in \mathbb{R}^{P \times C}$ 通过平均池化下采样为 $M$ 个尺度：$x_m \in \mathbb{R}^{\lfloor P/2^m\rfloor \times C}$。$x_0$（原始输入）保有最细粒度的时间变异，$x_M$ 侧重宏观趋势[^src-timemixer]。

### PDM（Past-Decomposable-Mixing）

PDM 是 TimeMixer 的核心创新。它不直接混合多尺度序列，而是先将每个尺度的序列分解为季节性（seasonal）和趋势（trend）分量（使用 Autoformer 的 SeriesDecomp），然后对两组分量分别进行不同方向的混合[^src-timemixer]：

1. **Seasonal Mixing（自下而上）**：细粒度的季节性细节逐层向上聚合，补充粗尺度的季节性建模——因为更大周期由更小周期组成（如周流量由日变化聚合）[^src-timemixer]。
2. **Trend Mixing（自上而下）**：粗尺度的宏观趋势逐层向下注入，引导细尺度的趋势提取——因为细粒度中的细节噪声会干扰宏观趋势的捕捉[^src-timemixer]。

混合操作均为时间维度上的两层线性变换 + GELU 激活，通过残差连接实现[^src-timemixer]。可视化分析显示：Seasonal Mixing 权重呈周期性变化，Trend Mixing 权重以对角局部聚合为主，验证了方向性设计的必要性[^src-timemixer]。

### FMM（Future-Multipredictor-Mixing）

每个尺度配备一个独立的单层线性预测器，将对应尺度的历史混合表示直接回归到未来 $F$ 步。所有尺度的预测结果求和集成：

$$\hat{x} = \sum_{m=0}^{M} \text{Predictor}_m(x_m^L)$$

Loss 基于集成结果计算（$\|x - \hat{x}\|$），而非对各预测器单独监督——这使得模型自动学习不同尺度预测器的互补能力[^src-timemixer]。可视化显示：细尺度预测器侧重捕捉季节性细节，粗尺度预测器侧重宏观趋势[^src-timemixer]。

## 性能表现

### 长期预测（8 benchmark，平均 4 个预测长度：96/192/336/720）

| 数据集 | TimeMixer MSE | PatchTST MSE | 相对提升 |
|--------|--------------|-------------|---------|
| Weather | 0.240 | 0.265 | -9.4% |
| Solar-Energy | 0.216 | 0.287 | -24.7% |
| Electricity | 0.182 | 0.216 | -15.7% |
| Traffic | 0.484 | 0.529 | -8.5% |
| ETTh1 | 0.447 | 0.516 | -13.4% |
| ETTh2 | 0.364 | 0.391 | -6.9% |
| ETTm1 | 0.381 | 0.406 | -6.2% |
| ETTm2 | 0.275 | 0.290 | -5.2% |

输入长度统一为 96[^src-timemixer]。

### 短期预测

- **M4**（6 个频率子集）：加权平均 SMAPE 11.723，MASE 1.559，OWA 0.840，全面超越 TimesNet、N-HiTS、N-BEATS、SCINet、PatchTST 等 14 个基线[^src-timemixer]。
- **PEMS**（4 个交通数据集）：在 PEMS03/04/07/08 上 MAE/MAPE/RMSE 均最优，证明在复杂多变量场景下仍有效——而 PatchTST、DLinear 等 channel-independence 模型在 PEMS 上退化严重[^src-timemixer]。

### 效率

GPU 内存和运行时间均优于 PatchTST，在输入长度 192–3072 范围内保持高效。全 MLP 架构不依赖注意力机制，避免了 $O(L^2)$ 的计算开销[^src-timemixer]。

### 消融实验

| 消融 | M4 SMAPE | PEMS04 MAE | ETTm1 MSE |
|------|----------|-----------|----------|
| 完整 TimeMixer ① | 11.723 | 19.21 | 0.390 |
| 去除 FMM ② | 12.503 (+6.7%) | 21.67 (+12.8%) | 0.402 (+3.1%) |
| 去除 Seasonal Mixing ③ | 13.051 (+11.3%) | 24.49 (+27.5%) | 0.411 (+5.4%) |
| 去除 Trend Mixing ④ | 12.911 (+10.1%) | 22.91 (+19.3%) | 0.405 (+3.8%) |
| 错误混合方向 ⑦ | 13.012 (+11.0%) | 22.27 (+15.9%) | 0.412 (+5.6%) |
| 完全去除 PDM ⑩ | 12.468 (+6.4%) | 24.87 (+29.5%) | 0.405 (+3.8%) |

每个组件都不可或缺，混合方向的选择（seasonal bottom-up、trend top-down）尤为关键[^src-timemixer]。

## 与其他模型的关系

- **[[autoformer|Autoformer]]**：TimeMixer 继承 Autoformer 的 SeriesDecomp 分解模块，但将其扩展到多尺度——在每个尺度上分别分解后再混合，而非仅做单一尺度的分解[^src-timemixer]。
- **[[fedformer|FEDformer]]**：FEDformer 的 [[moe-decomposition|MOEDecomp]] 扩展了 Autoformer 的分解；TimeMixer 直接使用基础 Moving-Average 版本，复杂度更低，将重心放在多尺度混合而非分解增强上[^src-timemixer]。
- **[[dlinear|DLinear]]**：DLinear 将分解作为预处理步骤（decomposition → linear regression），TimeMixer 将分解嵌入深度网络内部，且在多尺度上逐层混合，架构深度远大于 DLinear 的单层线性[^src-timemixer]。
- **[[patchtst|PatchTST]]**：PatchTST 使用 channel independence 和 patch tokenization，在 Traffic 等多变量数据集上退化严重；TimeMixer 不采用 CI，在多变量场景下更鲁棒[^src-timemixer]。
- **[[itransformer|iTransformer]]**：iTransformer 将 attention 用于变量维度、FFN 用于时间维度；TimeMixer 完全不使用 attention，在时间维度上用 MLP 混合，在长期预测上表现更优[^src-timemixer]。
- **[[timesnet|TimesNet]]**：TimesNet 通过 FFT 发现多周期、1D→2D reshape 后用 2D 卷积；TimeMixer 通过下采样获取多尺度、用线性混合取代卷积，计算更高效[^src-timemixer]。
- **[[cyclenet|CycleNet]]**：CycleNet 显式建模全局周期并将周期分量作为残差；TimeMixer 通过多尺度混合隐式建模周期性（seasonal mixing 的自下而上性质），但允许不同尺度呈现不同周期模式[^src-timemixer]。
- **[[lstf|LSTF]]**：TimeMixer 在 LSTF 的所有 8 个 benchmark 上均取得 SOTA，是该问题当前最优模型之一[^src-timemixer]。
- **[[hephestus|HEPHAESTUS]]**：HEPHAESTUS（ICLR 2026 审稿中）将 TimeMixer 的固定下采样多尺度混合替换为输入自适应的 [[ams-moe|AMS-MoE]] 动态路由，并通过 [[periodic-temporal-attention|PTA]] 显式建模日/周周期。两者共享多尺度混合的基本思想，但 HEPHAESTUS 以 MoE 路由取代固定分解策略[^src-timemixer]。
- **[[pir|PIR]]**：PIR（NeurIPS 2025）以 TimeMixer 为 channel-dependent 骨干测试后处理修订，论文报告 48 个设置平均 MSE 降 2.34%，作者将较小收益归因于 CD 模型已利用协变量信息[^src-pir]。

## 局限性

1. 线性混合层在超长输入时参数量增大，不利于移动端部署[^src-timemixer]。
2. 仅在时间维度混合，未涉及变量维度跨通道交互[^src-timemixer]。
3. 缺少对多尺度混合设计最优性和完备性的理论分析[^src-timemixer]。
4. 未来工作方向：探索 attention-based 或 CNN-based 混合以提升参数效率，以及引入变量维度的混合[^src-timemixer]。
5. 与 [[rstib-mlp|RSTIB-MLP]]（ICML 2025）同属 MLP-based 时序预测模型，但侧重点不同：TimeMixer 聚焦多尺度混合提升预测精度，RSTIB-MLP 聚焦 [[information-bottleneck|信息瓶颈]] 引导的鲁棒表示学习以对抗噪声扰动[^src-rstib]。

[^src-timemixer]: [[source-timemixer]]
[^src-rstib]: [[source-rstib-mlp]]
[^src-pir]: [[source-pir]]
