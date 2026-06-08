---
title: "Mixed Channel Dependency"
type: concept
tags:
  - time-series-forecasting
  - channel-dependency
  - diffusion-models
  - hybrid-architecture
  - multivariate
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Mixed Channel Dependency

**Mixed Channel Dependency**（混合通道依赖）是 [[middir|MiDDiR]] 提出的非对称架构策略：历史序列编码使用 Channel-Dependent（CD）方式以获取信息量丰富的跨通道表示，而去噪生成阶段使用 Channel-Independent（CI）方式以降低联合分布建模的复杂度[^src-middir]。

## 动机

多元时间序列预测始终面临 **CD vs CI 的权衡**[^src-middir]：

- **CD（Channel-Dependent）**：可捕获跨变量相关性，提供更丰富的历史表示，但高维联合分布建模复杂度高，参数量随通道数快速增长
- **CI（Channel-Independent）**：每个通道独立处理，降低建模复杂度和参数量，但丢失跨变量信息，对强相关变量的预测不足

以往工作要么选择全 CD（[[crossformer|Crossformer]]、[[itransformer|iTransformer]]），要么全 CI（[[patchtst|PatchTST]]、[[simdiff|SimDiff]]），或仅在浅层注入 CD（[[cvpe|CVPE]]）[^src-middir]。混合通道依赖的不同之处在于：**编码和去噪采用不同策略**，而非统一策略。

## 设计

```
输入 X^o (T × C)
    │
    ▼
┌──────────────────────┐
│  CD Encoder          │  通道依赖：FC + Multi-head Attention
│  ϕ(X^o) → e ∈ R^C×H │  跨通道信息混合
└──────┬───────────────┘
       │ e_c (每个通道独立)
       ▼
┌──────────────────────┐
│  CI Denoiser (×C)    │  通道独立：DiT block + AdaLN
│  ϵ_θ(x^p_n, n, e_c)  │  各通道独立去噪生成
└──────────────────────┘
```

### CD 编码器

L 层编码块，每层先做时间维度的全连接投影，再做跨通道的 Multi-head Attention[^src-middir]：

$$z^l = \text{GeLU}(e^{l-1} W^l + b), \quad e^l = \text{softmax}\left(\frac{z^l W^{lQ}(z^l W^{lK})^\top}{\sqrt{D}}\right)z^l W^{lV} + e^{l-1}$$

最终投影得到 H 维通道级隐向量 e ∈ R^(C×H)[^src-middir]。

### CI 去噪器

采用 DiT 类架构，每个通道独立去噪[^src-middir]：

1. **Patch Embedding**：输入序列划分为 patch（patch size p，重叠 p/2），作为 Transformer token
2. **DiT Block × N**：MHSA + MLP，AdaLN 注入扩散步嵌入和 CD 编码条件信号 ϕ(X^o)_c
3. **Unpatch + Linear Output**：恢复为噪声预测 ϵ̂

条件注入使用零初始化 AdaLN（与 [[dit|DiT]] 的 adaLN-Zero 一致）[^src-middir]。

## 有效性证据

MiDDiR 消融实验[^src-middir]：

| 配置 | ETTh1 MAE | ETTh1 CRPS | Traffic MAE | Traffic CRPS |
|------|-----------|------------|-------------|--------------|
| MiDDiR (full) | **0.431** | **0.300** | **0.269** | **0.198** |
| w/o RG | 0.432 | 0.301 | 0.277 | 0.207 |
| w/o CD | 0.434 | 0.300 | 0.298 | 0.231 |
| w/o CD & RG | 0.435 | 0.301 | 0.301 | 0.235 |

- 去掉 CD 编码在 Traffic（862 通道）上 MAE 从 0.269 升至 0.298（~10.8% 退化），说明高维场景下 CD 编码贡献显著
- 注意力图可视化：ETTh1 学习到清晰的跨通道依赖模式，Weather 仅学习到弱相关——编码器自适应利用可用依赖[^src-middir]

## 与其他 CI/CD 策略对比

| 策略 | 编码 | 去噪/预测 | 代表模型 |
|------|------|----------|---------|
| 全 CI | CI | CI | PatchTST, SimDiff, S-Mamba |
| 全 CD | CD | CD | Crossformer |
| 浅层 CD 注入 | CI+ | CI | CVPE |
| CD 空间模块 + CI 时间模块 | CI (时) | CI (时) | CPiRi |
| **混合 CD/CI** | **CD** | **CI** | **MiDDiR** |
| inverted (var-token) | - | CD-like attention | iTransformer |

## 相关概念

- [[channel-independence]] — CI 策略的全面分析
- [[cross-dimension-dependency]] — CD 建模的概念基础
- [[retrieval-guidance]] — 混合依赖的推理增强机制
- [[middir]] — 混合通道依赖的提出模型

[^src-middir]: [[source-middir]]
