---
title: "UniCA: Unified Covariate Adaptation for Time Series Foundation Model"
type: source-summary
tags:
  - time-series
  - foundation-model
  - covariate-adaptation
  - multimodal
  - iclr2026
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# UniCA 论文摘要

## 论文信息

- **标题**：UniCA: Unified Covariate Adaptation for Time Series Foundation Model
- **作者**：Lu Han*, Yu Liu*, Lan Li, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao Ma†, Han-Jia Ye†, De-Chuan Zhan
- **机构**：南京大学、蚂蚁集团
- **会议**：ICLR 2026 Poster
- **arXiv**：arXiv:2506.22039v2 (2025-06-27 提交，2026-03-24 修订)
- **代码**：https://github.com/hanlu-nju/UniCA

> *Equal Contribution, †Corresponding Author

## 论文信息

- **标题**：UniCA: Unified Covariate Adaptation for Time Series Foundation Model
- **作者**：Lu Han*, Yu Liu*, Lan Li, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao Ma†, Han-Jia Ye†, De-Chuan Zhan
- **机构**：南京大学、蚂蚁集团
- **会议**：ICLR 2026 Poster
- **arXiv**：arXiv:2506.22039v2 (2025-06-27 提交，2026-03-24 修订)
- **代码**：https://github.com/hanlu-nju/UniCA

> *Equal Contribution, †Corresponding Author

---

## 核心论点

### 问题背景

时间序列基础模型 (Time Series Foundation Models, TSFMs) 通过大规模预训练取得了显著成功。然而，它们的设计主要针对**单变量实数值序列**，这限制了它们处理一般预测任务的能力，这些任务通常涉及多样且往往是**异构的协变量**——如分类变量和多模态数据（图像、文本）[^src-unca]。

### 核心问题

1. **架构限制**：大多数最先进的 TSFMs（Chronos、MOMENT、TimesFM 等）本质上是为单变量预测设计的，以独立处理每个时间序列。这种架构选择使得无法利用目标序列与其外生协变量之间的关键关系。

2. **同质性假设**：标准 TSFM 预训练范式施加了一个关键限制——它只能有效利用**同质协变量**（与目标序列形式相同的实值时间序列）。这限制了处理**异构协变量**的能力。

3. **异构协变量的多样性**：
   - **结构化分类变量**：如商品 ID、门店位置、事件类型、日历特征
   - **多模态输入**：如图像（卫星云图、工业检测）、文本（新闻、天气报告）

### 解决方案

论文提出 **Unified Covariate Adaptation (UniCA)** 框架，包含两个核心组件：

1. **协变量同质化 (Covariate Homogenization)**：将异构协变量（分类变量、图像、文本）转换为统一的高阶同质时间序列表示

2. **双注意力融合机制**：
   - **Pre-Fusion 模块**：在编码前将历史协变量信息融入目标序列
   - **Post-Fusion 模块**：在编码后将未来已知协变量信息融入表示

3. **冻结骨干设计**：保持预训练的 TSFMs 参数不变，仅训练适配器模块，保留泛化能力[^src-unca]

---

## 核心贡献

### 贡献 1：问题形式化

将 TSFMs 适应到一般协变量感知预测场景的形式化定义。给定：
- 目标时间序列 $Y_{1:T} \in \mathbb{R}^{T \times 1}$
- 动态协变量 $C_{1:T+H}$（过去和未来）
- 静态协变量 $S$

预测目标：$\hat{Y}_{T+1:T+H} = f(Y_{1:T}, C_{1:T+H}, S)$

### 贡献 2：UniCA 框架

- **协变量同质化**：使用专用编码器（CNN/Transformer）提取异构特征，然后通过线性层投影为同质时间序列
- **双注意力融合**：Pre-Fusion（CAP 机制）+ Post-Fusion（自注意力融合）
- **参数高效**：冻结骨干，仅训练适配器模块

### 贡献 3：广泛实验

在 12 个单模态协变量数据集和多模态数据集（MMSP、Time-MMD）上验证了 UniCA 的有效性。

---

## 实验结果

### 单模态协变量数据集

| 模型 | MAPE | 相对基线 |
|------|------|----------|
| TimesFM (Zero-shot) | 0.598 | baseline |
| Chronos-Bolt (UniCA) | **0.506** | **-15.4%** |
| Time-MoE (UniCA) | 0.514 | -14.0% |

### 多模态数据集

**MMSP (图像-时序)**：
| 模型 | MAE | 提升 |
|------|-----|------|
| TimesFM (Zero-shot) | 0.778 | - |
| TimesFM (UniCA) | **0.634** | **-6.5%** |

**Time-MMD (文本-时序)**：
| 模型 | MAPE | 提升 |
|------|------|------|
| TimesFM (Zero-shot) | 0.900 | - |
| TimesFM (UniCA) | **0.656** | **-5.9%** |

---

## 关键发现

### 1. 适配器 vs 全参数微调

- 冻结骨干 + 适配器 (Frozen + Adapter) 在 MAPE 和 CRPS 上优于全参数微调
- 这验证了设计直觉：适配器模块是弥合通用时间序列表示与任务特定协变量上下文之间差距的轻量级但有效机制

### 2. 协变量同质化的泛化性

将协变量同质化器 (CH) 集成到专用模型（TFT、TiDE）中也带来显著提升：
- TFT + CH：MMSP 上 MAE 降低 5%，Time-MMD 上降低 13.0%
- TiDE + CH：MMSP 上 MAE 降低 30.1%，Time-MMD 上降低 10.5%

### 3. 同质化维度

- 从 $d_{het}=1$ 增加到 4 时性能显著提升（MSE 从 0.15 降至 0.10 以下）
- 超过 8 后收益递减
- 默认值 $d_{het}=4$ 是最佳折中[^src-unca]

---

## 局限性

1. **编码器选择**：异构协变量的编码器选择影响性能，当前使用简单 CNN/GIST
2. **维度敏感性**：默认同质化维度可能不适用于所有场景
3. **未来协变量依赖**：需要未来已知协变量信息

---

## 相关工作对比

| 方法 | 同质协变量 | 异构协变量 | 多模态 |
|------|-----------|-----------|--------|
| DeepAR | ✓ | ✗ | ✗ |
| TFT | ✓ | ✗ | ✗ |
| Moirai | 有限 | ✗ | ✗ |
| Chronos | ✗ | ✗ | ✗ |
| UniCA | ✓ | ✓ | ✓ |

---

## 引用

[^src-unca]: [[source-unca]]