---
title: "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"
type: source-summary
tags:
  - instaflow
  - rectified-flow
  - one-step-generation
  - diffusion-model
  - text-to-image
  - iclr-2024
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

**InstaFlow** 由 Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu (UT Austin / UIUC / Tsinghua) 提出，发表于 **ICLR 2024**（arXiv:2309.06380, Sep 2023）[^src-instaflow]。这是首个成功将大规模 Stable Diffusion 蒸馏为高质量一步文生图模型的工作，核心洞见是：**直接蒸馏 SD 完全失败（FID=40.9 vs SD 的 22.8），但经过 Rectified Flow reflow 拉直轨迹后再蒸馏，一步模型的质量飞跃到可用水平（FID=23.4）**。

PDF 路径: `Zotero/storage/9F7GKVSH/`。

## 核心贡献

1. **证明 reflow 是蒸馏成功的必要前提** — 直接蒸馏 SD 在网格搜索 9 组超参数后最佳 FID-5k=40.9（vs 老师 SD 22.8，gap=18.1），而 2-Rectified Flow+Distill gap 缩小到 8.9（31.0 vs 22.1），reflow 将蒸馏困难度减半[^src-instaflow]
2. **首次将 Rectified Flow pipeline 扩展到十亿参数级别** — 从 SD 1.4/1.5 预训练 U-Net (0.9B) 开始微调而非从零训练，reflow+distill 总成本 ~199 A100 GPU days（vs SD 从头训练的 ~6250 天）[^src-instaflow]
3. **提出了文本条件的 Rectified Flow 公式** — 将 Rectified Flow 扩展到文本条件生成：$v_{k+1} = \arg\min \mathbb{E}[ \int_0^1 \|(X_1-X_0) - v(X_t,t|T)\|^2 dt]$，其中 $X_1 = \text{ODE}[v_k](X_0|T)$，并设计了推理时的 CFG 等效机制 $v^\alpha = \alpha \cdot v(\cdot|T) + (1-\alpha) \cdot v(\cdot|\text{NULL})$[^src-instaflow]
4. **Stacked U-Net 架构创新** — 通过串联两个 U-Net 并裁剪冗余模块（删除首个 U-Net 的部分下采样/上采样/中间块），得到 1.7B 参数的 InstaFlow-1.7B，推理仅 0.12 秒[^src-instaflow]

## 方法流程

### 三步管线
1. **取预训练 SD 作为 1-flow**: $v_1 = v_{\text{SD}}$（通过 DPM-Solver 25 步实现）
2. **文本条件 reflow**: 用 SD 的 ODE 端点 $(X_0, \text{ODE}[v_1](X_0|T))$ 作为训练数据，线性插值 $X_t = tX_1 + (1-t)X_0$，训练 $v_2$
3. **蒸馏**: 从 $v_k$ 做路径蒸馏，结合 L2（大批量 21.5K 步）和 LPIPS（精细调优 18K 步）两阶段损失

### 训练阶段（InstaFlow-0.9B）
| 阶段 | Batch | 步数 | Loss | GPU Days |
|------|-------|------|------|----------|
| Reflow 小batch | 64 | 70K | L2 | 11.2 |
| Reflow 大batch | 1024 | 25K | L2 | 64 |
| Distill L2 | 1024 | 21.5K | L2 | 54.4 |
| Distill LPIPS | 1024 | 18K | LPIPS | 53.6 |

数据生成开销 ~16 天（离线生成配对数据，无需在线运行教师模型）[^src-instaflow]。

### 为什么直接蒸馏 SD 失败？

论文给出了三方面证据[^src-instaflow]：

1. **轨迹弯度** — SD 的 $S(Z)$（速度偏离度）高，像素轨迹弯弯曲曲；2-RF 的 $S(Z)$ 大幅下降，像素轨迹接近直线（Figure 6）
2. **配对规律性** — 同一噪声下，SD+Distill 生成的图和老师 SD 完全不相似；2-RF+Distill 生成的图和老师 2-RF 高度相似（Figure 5），说明 reflow 让配对更规律
3. **师生差距缩小** — SD→SD+Distill: 22.8→40.9（差 18.1）；2-RF→2-RF+Distill: 22.1→31.0（差 8.9）

## 实验结果

### COCO 2017-5k
| 模型 | FID-5k | CLIP | 推理时间 | 备注 |
|------|--------|------|---------|------|
| SD 1.5 (25步 DPM-Solver) | ~20.1 | — | 0.88s | 老师模型 |
| PD-SD (1步) | 37.2 | — | 0.09s | Progressive Distillation |
| 2-RF (25步) | 21.5 | — | 0.88s | reflow保持边际分布 |
| InstaFlow-0.9B (1步) | **23.4** | 0.304 | 0.09s | 基于 SD 1.5 |
| InstaFlow-1.7B (1步) | **22.4** | 0.309 | 0.12s | Stacked U-Net |

### COCO 2014-30k — 与全球模型横向对比
| 模型 | FID-30k | 推理时间 | 范式 |
|------|---------|--------|------|
| Imagen (3B) | 7.27 | — | Diffusion |
| Parti-3B | 8.10 | — | AR |
| GigaGAN (1B) | 9.09 | 0.13s | GAN |
| StyleGAN-T (1B) | 13.90 | 0.10s | GAN |
| **InstaFlow-0.9B** | **13.10** | **0.09s** | **Diff-Distill** |
| InstaFlow-1.7B | 11.83 | 0.12s | Diff-Distill |

InstaFlow-0.9B 是首个以纯监督学习（无对抗训练、无 RL）与 GAN 匹敌的一步扩散模型[^src-instaflow]。

### 少步推理对比
在 {1,2,4,8} 步与 SD 1.5 DPM-Solver 对比：N≤4 时 2-RF 全面领先；N=1 时 SD 出纯噪声而 2-RF 出模糊但有结构的图；N=8 时接近但 2-RF 仍稍好[^src-instaflow]。

### 引导力度
CFG α 的最佳平衡点 ≈1.5-2.0，远低于 SD 的 5-7.5——拉直轨迹需要更少引导[^src-instaflow]。

## 局限与后续影响

### 论文承认的限制
1. Reflow 未完全收敛（2-RF 25步 FID=21.5 不及原始 SD 1.5 的 20.1），更长训练可能进一步改善[^src-instaflow]
2. 复杂组合提示仍有失败案例（属性错乱、物体关系不对）[^src-instaflow]
3. 引导力度 α 是敏感超参，无自适应选择方案[^src-instaflow]
4. 多次 reflow 边际收益递减（k=2→3 从 31.0→29.3）[^src-instaflow]
5. 仅在 SD U-Net 架构验证，未测试 DiT 等非 U-Net 架构[^src-instaflow]
6. 未分析一步模型的多样性指标（Recall, Precision, Coverage），模式崩塌风险未回答[^src-instaflow]

### 影响
InstaFlow 首次证明了 Rectified Flow pipeline 在大规模生成模型上的可行性，直接促成了 SD3、FLUX 等工业级模型采用 flow matching / rectified flow 作为训练范式[^src-instaflow]。与 ControlNet 的兼容性（无需任何微调即可直接使用预训练 ControlNet 权重）更是一个惊喜发现，说明 reflow+distill 不破坏 SD 潜空间的语义结构[^src-instaflow]。

[^src-instaflow]: [[source-instaflow]]
