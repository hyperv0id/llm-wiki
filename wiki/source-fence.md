---
title: "FENCE: Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation"
type: source-summary
tags:
  - diffusion-models
  - spatiotemporal-imputation
  - dynamic-guidance
  - traffic
  - classifier-free-guidance
  - aaai-2026
created: 2026-06-08
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Source: FENCE

**作者**: Xiaowei Mao, Huihu Ding, Yan Lin, Tingrui Wu, Shengnan Guo, Dazhuo Qiu, Feiling Fang, Jilin Hu, Huaiyu Wan (北京交通大学, Aalborg University, 中国地质大学, 华东师范大学)
**发表**: AAAI 2026 会议论文（Proceedings pp. 15528–15536）；arXiv:2601.04572（2026-01 v1，含附录推导）
**代码**: https://github.com/maoxiaowei97/FENCE
**raw 版本**: arXiv v1（`2601.04572.pdf`）与 AAAI-26 会议版（`fence-spatial-temporal-feedback-diffusion-guidance-aaai26.pdf`），两者正文内容一致，会议版无附录。

## 核心论点

FENCE 提出了一个面向扩散模型时空交通数据插补的动态反馈引导机制，核心贡献在于解决现有扩散插补方法（如 [[csdi|CSDI]]、[[pristi|PriSTI]]）中固定引导尺度（guidance scale）的不足[^src-fence]。现有方法对所有节点、所有去噪步使用统一的引导尺度 $\lambda$，但这一策略在高缺失率节点上失效：稀疏观测值提供的条件信息不足，导致生成过程漂移到无条件先验分布，而非遵守条件观测[^src-fence]。FENCE 的回应是：将引导尺度变为去噪步 $k$ 和样本 $x_k$ 的函数，通过后验似然 $p(c|x_k)$ 的近似估计动态调整引导强度——后验降低时增大引导，后验升高时降低引导以避免过校正[^src-fence]。

## 方法

### 后验驱动的动态引导缩放

FENCE 采用加性误差假设（论文标注该公式采用自 Koulischer et al. 2025 的 Feedback Guidance 工作[^src-fence]）：学习的条件分布 $p_{\theta,k}(x_k|c)$ 是真实条件分布 $p_k(x_k|c)$ 和真实无条件分布 $p_k(x_k)$ 的线性组合，权重 $\pi \in [0,1]$ 表示对条件模型学习效果的事先置信度[^src-fence]。通过推导，引导尺度直接表达为后验似然的函数：

$$\lambda(x_k, k) \approx \frac{p_{\theta,k}(c|x_k)}{p_{\theta,k}(c|x_k) - (1-\pi)}$$

当后验似然 $p(c|x_k)$ 高时，$\lambda \to 1$（轻度引导）；当后验接近阈值 $(1-\pi)$ 时，$\lambda$ 急剧增大，施加更强引导[^src-fence]。

### 后验似然估计

FENCE 通过追踪扩散反向马尔可夫链来估计后验似然（论文注明该方法受 Koulischer et al. 2025 启发[^src-fence]）：

$$\log p_{\theta,k-1}(c|x_{k-1}) = \log p_{\theta,k}(c|x_k) + \log p_\theta(x_{k-1}|x_k, c) - \log p_\theta(x_{k-1}|x_k)$$

引入温度 $\tau$ 和偏移 $\delta$ 两个超参数控制更新幅度和激活时机[^src-fence]。

### 聚类感知引导

不同节点对条件观测的符合程度不同，全局统一引导尺度次优。FENCE 利用空间注意力分数 $A_{\text{attn}}$ 在每步去噪时对节点做 k-means 聚类，对每个聚类 $C_j$ 计算聚类级对数后验的均值，得到更稳定的引导尺度估计[^src-fence]。

### 两阶段训练

为避免无条件先验学习干扰条件插补，FENCE 采用两阶段训练：先训练无条件生成模型学习先验 $p_\theta(x)$，收敛后保存权重作为初始化；再在条件观测上微调该模型（权重继续更新）以学习条件分布[^src-fence]。

## 关键结果

- 在 PEMS04、PEMS07、PEMS08 三个数据集上，两种缺失模式（SR-TC 和 SC-TC，80% 缺失率）下全面超越 [[csdi|CSDI]]、[[pristi|PriSTI]]、[[imputeformer|ImputeFormer]] 等 8 个基线，MAPE 平均提升 6.26%[^src-fence]
- 在最具挑战性的 SC-TC（时空聚类缺失）场景下优势尤为显著[^src-fence]
- 消融实验验证了反馈引导（wo-F）和聚类感知（wo-C）两个组件的必要性[^src-fence]
- $\pi=0.5$ 时性能最佳，聚类数 $N/20$ 时最优[^src-fence]

## 贡献（论文自述）

1. 提出 FENCE，一种时空反馈扩散引导方法，在生成过程中动态控制引导尺度，实现对缺失交通数据的高保真插补[^src-fence]
2. 提出聚类感知引导机制，利用时空相关性为每个节点计算更准确的定制化引导尺度[^src-fence]
3. 大量实验表明 FENCE 显著提升真实世界时空交通数据集上的插补精度[^src-fence]

注：论文将加性误差公式标注为采用 Koulischer et al. 2025 的 Feedback Guidance 工作，后验似然估计方法亦受其启发；FENCE 的定位是将该反馈引导机制应用于时空交通插补场景并增加聚类感知设计。

## 局限性

- 论文实验仅在 PEMS04、PEMS07、PEMS08 交通数据集上验证，其他时空数据类型（如空气污染、气象）的泛化性未在论文中报告[^src-fence]

## 相关工作引用口径

FENCE 的 Related Work 将判别式插补模型举例为 "(Cao et al. 2018; Che et al. 2018; Weng et al. 2025)"，其中 Weng et al. 2025 即 [[lets-group|Let's Group]]（IJCAI-25，参考文献著录 pp. 3471–3479）[^src-fence]。需要注明的口径差异：Let's Group 原文的任务设定与实验均为交通预测（可学习子图划分以降低 STGNN 内存开销），未包含插补实验；「判别式插补模型」是 FENCE 相关工作段落中的归类，非 Let's Group 原文的自我定位[^src-lets-group]。详见 [[source-lets-group]]。

[^src-fence]: [[source-fence]]
[^src-lets-group]: [[source-lets-group]]