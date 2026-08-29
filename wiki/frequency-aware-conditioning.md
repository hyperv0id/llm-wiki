---
title: "频率感知条件化"
type: concept
tags:
  - flow-matching
  - frequency-domain
  - conditioning
  - image-generation
created: 2026-05-09
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# 频率感知条件化（Frequency-Aware Conditioning）

**频率感知条件化**是一种在生成模型中显式利用频域信息作为条件信号的建模范式。其核心理念是：生成模型在空域操作时天然对频率成分的恢复是非均匀的，通过显式的频率条件引导可以使模型更好地保留全局结构和局部细节。该范式同时出现在时序插补任务中（见下节 FGTI）[^src-fgti]。

## 动机

标准流匹配（Flow Matching）和扩散模型在潜域中均匀注入高斯噪声，但噪声对频域各分量的影响是不均匀的。实验观察发现：[^src-freqflow]

- **低频成分**（全局形状、颜色分布）在生成早期迅速重建
- **高频成分**（纹理、边缘）在生成晚期才逐步细化
- 缺乏显式频率引导时，模型倾向于产生轻微模糊或平滑的结果（尤其在高频区域）

这一现象揭示了根本性的 gap：生成模型在空域操作，但噪声损坏和恢复过程在频域具有非均匀特性——然而频域特性既没有被显式建模也没有被有效利用。[^src-freqflow]

## 通用形式

频率感知条件化可以形式化为：

给定噪声图像 $X_t$，提取频域条件信号 $h_t$ 并注入生成模型 $f_\theta$：

$$
h_t = \mathcal{G}(\mathcal{F}_{\text{filter}}(X_t))
$$

$$
\hat{Y}_t = f_\theta(X_t, h_t, t, c)
$$

其中 $\mathcal{F}_{\text{filter}}$ 为频域滤波/分解操作，$\mathcal{G}$ 为条件编码网络。[^src-freqflow]

## FreqFlow 中的实现

[[freqflow|FreqFlow]] 是频率感知条件化的代表性实现，使用双分支架构：

1. **频率分支**（条件提取器 $\mathcal{G}$）：通过 DFT → 高斯滤波 → IDFT 分解噪声图像为低/高频分量，用统一 ViT 处理，输出融合特征 $h_t$
2. **空间分支**（生成器 $f_\theta$）：接收 $X_t$ 和 $h_t$，通过逐元素加法融合后生成图像

自适应权重 $\omega_t = \sigma(\text{MLP}(h_t^L, h_t^H, t))$ ��现时变的频率条件化。[^src-freqflow]

## 时序插补中的实例：FGTI

[[fgti|FGTI]]（NeurIPS 2024）把该范式用于多变量时间序列插补：high-frequency filter（保留截止频率 F 以上分量，用于引导残差项插补）与 dominant-frequency filter（取幅值 top-κ 分量，提供趋势/季节项背景结构）从观测序列提取两组频域条件，经 Transformer 编码后由 time-frequency 与 attribute-frequency 两个 cross-attention 模块（Q、K 来自频域表示、V 来自时域隐表示）融入 DDPM 式去噪网络；论文并以条件熵命题（Prop 3.1）论证加入频域条件严格降低扩散反向过程的不确定性[^src-fgti]。

与 FreqFlow 的差异：条件信号从"噪声图像的频域分解"[^src-freqflow]变为"观测数据的频域分解"（FGTI 的两组条件均提取自观测序列，不来自加噪中间态 X̂_t）[^src-fgti]；注入方式从逐元素加法[^src-freqflow]变为 cross-attention[^src-fgti]；任务从图像生成变为条件插补。

## 与其他条件化范式的关系

| 范式 | 条件信号 | 注入方式 | 效果 |
|-----|---------|---------|------|
| 类别条件化 | 类别标签 $c$ | AdaIN/Cross-Attention | 控制生成类别 |
| 文本条件化 | CLIP embedding | Cross-Attention | 语义引导 |
| **频率条件化** | 频域分解特征 $h_t$ | 逐元素加法 | 控制全局结构/局部细节平衡 |
| 时间步条件化 | $t$ 嵌入 | AdaIN/SiLU | 控制噪声水平 |

频率条件化与其他条件化是正交的，可以组合使用。FreqFlow 同时使用了类别条件和时间步条件。[^src-freqflow]

## 优点

- 显式控制"先粗后细"的生成轨迹
- 降低高频误差（FreqFlow 高频误差 0.48 vs SiT 0.69）
- 加速训练收敛（更快达到低频目标）
- 提升生成图像的结构一致性和细节锐度

## 链接

- [[freqflow]] — FreqFlow，频率感知条件化的代表性实现
- [[fgti]] — FGTI，该范式在时序插补中的实例（NeurIPS 2024）
- [[diffusion-frequency-domain-theory]] — 频域理论统一视角（问题演化叙事）

## 引用

[^src-freqflow]: [[source-freqflow]]
[^src-fgti]: [[source-fgti]]
