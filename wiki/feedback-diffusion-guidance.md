---
title: "反馈扩散引导"
type: technique
tags:
  - diffusion-models
  - dynamic-guidance
  - classifier-free-guidance
  - posterior-likelihood
  - neurips-2025
  - aaai-2026
created: 2026-06-08
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# 反馈扩散引导 (Feedback Guidance, FBG)

**反馈扩散引导**（FeedBack Guidance, FBG）是一种状态与时间相关的动态扩散引导技术，由 Koulischer et al. 提出并发表于 NeurIPS 2025（图像生成场景）[^src-fbg]。其核心是把引导尺度 $\lambda$ 从固定超参数变为去噪状态 $x_t$ 与时刻 $t$ 的函数：用模型自身预测估计当前样本的条件符合度（后验似然 $p(c|x_t)$），后验降低时增大引导，后验升高时减小引导[^src-fbg]。[[fence|FENCE]]（AAAI 2026）随后将该机制引入时空交通数据扩散插补，并加入聚类感知设计[^src-fence]。

## 误差模型视角：加性 vs 乘性

论文提出把各种引导公式理解为"对学习到的条件分布的系统性误差做反转"的结果：由于联合训练中每个类别/prompt 的条件训练信号占比低，学到的条件分布会被无条件分布污染[^src-fbg]。

- **CFG 的隐式乘性假设**：将 CFG 公式反向改写（记 $\gamma = 1/\lambda$）可得 $p_{\theta,t}(x_t|c) \propto p_t(x_t)^{1-\gamma} p_t(x_t|c)^{\gamma}$，即学到的条件分布是真实条件分布与无条件分布的乘性混合[^src-fbg]。
- **FBG 的加性假设**：$p_{\theta,t}(x_t|c) = (1-\pi)p_t(x_t) + \pi p_t(x_t|c)$，其中 $\pi \in [0,1]$ 表示对条件模型学习效果的事先置信度[^src-fbg][^src-fence]。

论文认为加性假设比乘性假设限制更少：联合训练与训练对含多元素使得学到的条件分布在真实条件分布为零的区域非零，乘性假设无法表达这种重叠[^src-fbg]。附录 B 进一步论证：乘性混合与加噪操作不可交换（乘法与卷积不可交换），因此按 CFG 得分采样并不对应直观的锐化分布；加性混合与卷积可交换，在精确后验可得时能恢复预定义的边缘分布[^src-fbg]。

## 引导尺度公式

在加性假设下（并设无条件模型已学好），从 $p_t(x_t|c) \propto p_{\theta,t}(x_t|c) - (1-\pi)p_{\theta,t}(x_t)$ 出发，用链式法则可得与 CFG 形式相同的引导方程，但尺度是状态与时间的函数（原文 Eq. 7–8）[^src-fbg]：

$$\nabla_{x_t} \log p_t(x_t|c) = \nabla_{x_t} \log p_{\theta,t}(x_t) + \lambda(x_t,t)\big(\nabla_{x_t}\log p_{\theta,t}(x_t|c) - \nabla_{x_t}\log p_{\theta,t}(x_t)\big)$$

$$\lambda(x_t,t) = \frac{q}{q-(1-\pi)}, \qquad q = \frac{p_{\theta,t}(c|x_t)}{p_{\theta,t}(c)}$$

- 后验比 $q$ 高（预测已符合条件）→ $\lambda \to 1$，轻引导
- $q \to (1-\pi)$ → $\lambda$ 发散，强引导；$\pi$ 越大表示条件模型学得越好，引导只在后验很低时才激活[^src-fbg]
- 论文指出当 $0 < p_{\theta,t}(c|x_t) < 1-\pi$ 时 $\lambda$ 为负，并论证连续情形下不会跨越渐近线；离散化时将后验截断在 $p_{\min}$（等价于封顶 $\lambda_{\max}$）[^src-fbg]

> [!note] 与 FENCE 引用形式的差异
> 原文公式以后验与条件先验之比 $q = p_{\theta,t}(c|x_t)/p_{\theta,t}(c)$ 代入；[[fence|FENCE]] 论文与 wiki 早期版本采用的简写形式直接以 $p(c|x_k)$ 代入[^src-fence]。

## 后验似然追踪

后验 $p_{\theta,t}(c|x_t)$ 一般不可直接获取。FBG 沿用同组 Dynamic Negative Guidance 工作的思路，在去噪过程中追踪扩散马尔可夫链，利用反向转移的马尔可夫性质迭代更新对数后验（原文 Eq. 9–11）[^src-fbg][^src-fence]：

$$\log p_{\theta,t}(c|x_t) = \log p_{\theta,t+1}(c|x_{t+1}) - \frac{\tau}{2\sigma^2_{t|t-1}}\Big(\|x_t - \mu_{\theta,t}(x_{t+1}|c)\|^2 - \|x_t - \mu_{\theta,t}(x_{t+1})\|^2\Big) - \delta$$

高斯转移假设下，更新项即条件与无条件预测均值的加权 L2 距离差；所需量在去噪中均已算出，额外计算开销可忽略[^src-fbg]。

- **自我参照偏差与偏移 $\delta$**：若直接以条件模型自己的预测作为评估轨迹，后验会被人为抬高、抑制引导激活；线性偏移 $-\delta$ 迫使后验在扩散早期下降，使引导得以激活[^src-fbg]。
- **$\tau, \delta$ 重参数化**：论文将二者改写为两个归一化时间参数——$t_0$（引导尺度达到参考值 $\lambda_{\text{ref}}=3$ 的时刻）与 $t_1$（欧氏项与偏移项量级相当、引导开始回落的时刻）——以降低调参难度[^src-fbg]。

## 开环与闭环控制

论文用控制论语言定位方法：CFG 与 LIG（Limited Interval Guidance，在预定义噪声区间内施加引导）都是开环控制——引导尺度与当前状态无关；FBG 是闭环反馈控制，用自身预测的质量估计反馈调节引导量[^src-fbg]。作者称状态相关引导此前仅在负引导（negative guidance）方向有理论最优性结果（同组 DNG，ICLR 2025；Kim et al. 2025），正向引导只有启发式方案（如 SEGA）；论文自称是首个从第一性原理导出状态与时间相关引导尺度的工作[^src-fbg]。

## 实验证据（作者报告）

- **类别条件生成**：ImageNet 512×512、EDM2-XS、64 NFE，随机采样器与二阶 Heun PFODE 两种设定，指标为 FID / FD_DinoV2 / Precision-Recall，基线 CFG、CFG++、LinCFG（线性权重调度）、LIG 均按各自最优超参调优（表 1）。随机采样器下：FID 最优设定 FBG_pure 3.76 vs CFG 5.00（LIG 3.59）；FD_DinoV2 最优设定 FBG_pure 89.0 vs CFG 100.2（LIG 88.5）。组合 FBG_LIG 在 FD_DinoV2 上优于两个组成部分[^src-fbg]。
- **质量-多样性权衡**：CFG 提高质量但大幅牺牲多样性；LIG 因引导区间窄而保留多样性；FBG 达到与 CFG 相当的 Recall，同时 Precision 明显更高（图 3b）[^src-fbg]。
- **文本到图像**：Stable Diffusion 2（实现 VE 调度）、MS-COCO 3k prompts：FBG_pure FID 18.63 vs CFG 19.64，FD_DinoV2 53.11 vs 54.56，Aesthetic 5.75 vs 5.65（表 2）。论文明确说明该实验目的不是证明超越 CFG/LIG，而是展示可行性[^src-fbg]。
- **提示词与轨迹特异性**：自建 60 条、四档难度（memorized/basic/intermediate/hard）prompt 数据集：越难的 prompt 平均获得越强引导，memorized prompt 的引导尺度接近 1（图 4）；同一 prompt 下不同轨迹获得的引导量也不同（图 5）[^src-fbg]。

## 与 FENCE 的关系

[[fence|FENCE]]（AAAI 2026）将 FBG 的引导尺度公式与马尔可夫后验追踪引入时空交通数据插补（条件 $c$ 为观测值），并增加聚类感知引导（按空间注意力分数做 k-means 聚类，聚类内共享引导尺度）与两阶段训练[^src-fence]。两点差异值得注意：

1. **公式形式**：FENCE 采用省略 $p_{\theta}(c)$ 归一化的简写（见上文 callout）[^src-fence]。
2. **$\pi$ 取值**：FBG 论文在 ImageNet 上扫 $\pi \in \{0.999, 0.9999, 0.99999\}$、T2I 用 0.85–0.9，且结果对 $\pi$ 不敏感；FENCE 报告 $\pi=0.5$ 最佳[^src-fbg][^src-fence]。

## 局限性（论文自述）

- 加性混合模型基于数学简单性选择，未必贴近训练后模型的真实系统偏差；先验选择同样基于简单性[^src-fbg]
- 实验仅用 EDM2-XS 规模模型，更大架构（EDM2-L、DiT）与更大规模 T2I 评测（如 LAION-5B 级 prompt 集）留待后续[^src-fbg]
- 论文认为 MS-COCO prompts 均匀、偏简单，不是比较引导方法的理想基准[^src-fbg]

## 另见

- [[classifier-free-guidance]] — 固定尺度引导的原始形式
- [[fence]] — FBG 在时空插补中的应用与扩展（聚类感知引导）

[^src-fbg]: [[source-feedback-guidance-diffusion-models-arxiv25]]
[^src-fence]: [[source-fence]]
