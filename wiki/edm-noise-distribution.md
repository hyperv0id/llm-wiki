---
title: "EDM Noise Distribution"
type: technique
tags:
  - diffusion
  - training
  - noise
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# EDM Noise Distribution

EDM 噪声分布是 Karras 等人提出的训练噪声水平采样策略[^src-edm]，使用对数正态分布而非均匀分布来聚焦关键噪声水平。

## 分布定义

$$\ln(\sigma) \sim \mathcal{N}(\mu = -1.2, \sigma = 1.2)$$

即 $\sigma$ 服从对数正态分布，均值约为 0.44，标准差范围覆盖 0.002 到 80。

## 设计理由

EDM 分析了各噪声水平下的训练损失[^src-edm]：
- **低噪声区域**（σ → 0）：噪声成分极小，难以学习且不关键
- **高噪声区域**（σ → ∞）：目标始终趋近数据集均值
- **中间噪声区域**：包含可学习的结构化信息

因此使用对数正态分布将训练重点聚焦于中间噪声水平。

## 与其他方法的对比

| 方法 | 噪声分布 |
|------|----------|
| DDPM | σ ~ U[0, 1] |
| NCSN | σ = σ_j, j ~ U{0, M-1} |
| EDM | ln(σ) ~ N(-1.2, 1.2²) |

## 链接

- [[edm]] — EDM 论文
- [[diffusion-model]] — 扩散模型基础
- [[edm-preconditioning]] — 预处理技术

[^src-edm]: [[source-edm]]
