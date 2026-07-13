---
title: "无分类器引导"
type: technique
tags:
  - diffusion
  - conditional-generation
  - guidance
created: 2026-04-28
last_updated: 2026-07-13
source_count: 7
confidence: high
status: active
---

# 无分类器引导

无分类器引导（Classifier-Free Guidance, CFG）由 Ho & Salimans (2022) 提出，是一种不需要独立分类器的扩散模型条件生成技术。它联合训练条件模型和无条件模型，在推理时通过对两者的得分估计进行线性插值来实现条件控制。[^src-classifier-free-diffusion-guidance]

## 核心思想

CFG 的动机源于对[[classifier-guidance|分类器引导]]（Dhariwal & Nichol, 2021）的三点质疑：[^src-classifier-free-diffusion-guidance]

1. **额外训练成本**：分类器必须训练在噪声数据上，无法复用预训练分类器[^src-classifier-free-diffusion-guidance]；
2. **对抗性疑虑**：分类器引导等同于用梯度对抗攻击欺骗分类器，可能只是对抗性地提升基于分类器的指标（FID、IS）[^src-classifier-free-diffusion-guidance]；
3. **与 GAN 相似性**：沿分类器梯度方向更新生成器近似 GAN 训练，可能只是复用 GAN 对分类器指标的优势[^src-classifier-free-diffusion-guidance]。

CFG 的解决方案：训练**单一扩散模型**同时参数化条件模型 $\epsilon_\theta(x_t, c)$ 和无条件模型 $\epsilon_\theta(x_t, \varnothing)$。训练时以概率 $p_\text{uncond}$ 随机丢弃条件信息，使模型同时在两种模式下工作。[^src-classifier-free-diffusion-guidance]

## 数学形式

CFG 使用修改后的得分估计进行采样：[^src-classifier-free-diffusion-guidance]

$$
\tilde{\epsilon}_\theta(z_\lambda, c) = (1 + w)\,\epsilon_\theta(z_\lambda, c) - w\,\epsilon_\theta(z_\lambda)
$$

其中 $w \geq 0$ 是**引导强度**。$w=0$ 退化为标准条件生成，$w>0$ 放大条件影响。注意这与常见的形式 $\epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ 等价（令 $w = w'+1$）。

### 隐式分类器解释

公式受隐式分类器 $p^i(c|z_\lambda) \propto p(z_\lambda|c)/p(z_\lambda)$ 的启发。若得分估计是精确的保守向量场（即存在标量势），则 $\epsilon^*(z_\lambda, c) - \epsilon^*(z_\lambda)$ 正比于该隐式分类器的梯度。但实际中神经网络产生的得分不一定是保守场，因此 CFG 的修正方向**不一定对应任何分类器的梯度**——这证明 CFG 不是对分类器的对抗攻击，而是纯生成方法。[^src-classifier-free-diffusion-guidance]

### 联合训练

训练时以概率 $p_\text{uncond}$ 将条件 $c$ 设为空标记 $\varnothing$。Ho & Salimans 实验发现 $p_\text{uncond} \in \{0.1, 0.2\}$ 表现相当且显著优于 $0.5$——只需将少量模型容量分配给无条件任务即可获得有效引导信号。[^src-classifier-free-diffusion-guidance]

## 引导强度 $w$ 的行为

- **$w = 0$**：退化为标准条件生成，相当于直接使用 $\epsilon_\theta(x_t, c)$。[^src-classifier-free-diffusion-guidance]
- **$w > 0$**：放大条件信号的影响，降低多样性同时提高样本保真度。在 ImageNet 上，$w=0.1\sim0.3$ 取得最优 FID，$w\geq4$ 取得最优 IS——随着 $w$ 单调增加，FID 单调下降、IS 单调上升，形成清晰的保真度-多样性权衡曲线。[^src-classifier-free-diffusion-guidance]
- **典型取值**：文本到图像生成中 $w$ 通常在 $3\sim15$（等价于常见形式中的 $w'=2\sim14$）；类别条件 ImageNet 中 $w=1.5$ 左右。[^src-classifier-free-diffusion-guidance][^src-dit]

随着 $w$ 增大，模型生成结果更严格遵循条件，颜色趋向饱和，但样本多样性下降。[^src-classifier-free-diffusion-guidance]

## 原始基准结果（ImageNet, Ho & Salimans 2022）

| 设置 | $w$ | FID (↓) | IS (↑) |
|------|-----|---------|--------|
| 64×64, $p_\text{uncond}=0.1$ | 0.0 | 1.80 | 53.71 |
| 64×64, $p_\text{uncond}=0.1$ | 0.3 | **1.55** | 66.11 |
| 64×64, $p_\text{uncond}=0.1$ | 4.0 | 24.83 | **250.4** |
| 128×128, $T=256$ | 0.0 | 7.27 | 82.45 |
| 128×128, $T=256$ | 0.3 | **2.43** | 158.47 |
| 128×128, $T=256$ | 4.0 | 21.53 | **421.03** |

128×128 在 $w=0.3$ 时 FID **超越**分类器引导的 ADM-G（FID=2.97）；在 $w=4.0$ 时同时超越 BigGAN-deep 的最佳 IS 水平。[^src-classifier-free-diffusion-guidance]

## 与分类器引导的对比

| 特性 | 分类器引导 | 无分类器引导 |
|------|-----------|-------------|
| 是否需要额外分类器 | 是，需训练噪声鲁棒分类器 | 否，仅需扩散模型本身 |
| 训练复杂度 | 需额外训练分类器 | 只需随机丢弃条件（一行代码） |
| 推理计算量 | 需计算分类器梯度 | 需两次前向传播（条件+无条件） |
| 对抗性疑虑 | 是（分类器梯度=对抗攻击） | 否（得分不一定保守，无可解释的分类器） |
| 实际效果 | 受限于分类器质量 | 更优，广泛用于主流模型 |

CFG 已完全取代分类器引导成为扩散模型条件生成的标准技术。[^src-classifier-free-diffusion-guidance]

## 直觉解释：降低无条件似然

CFG 的效果可直觉理解为：在增大条件似然 $p(x|c)$ 的同时**降低无条件似然** $p(x)$，从而将生成推向条件分布的高密度区域而远离边缘分布的典型样本。这一"降低无条件似然"的负得分项（$-w\,\epsilon_\theta(z_\lambda)$）在此前文献中未被探索，可能有其他应用。[^src-classifier-free-diffusion-guidance]

## 应用

- **文本到图像生成**：CFG 是 Stable Diffusion、DALL-E 2、Imagen 等模型的核心组件，用于控制文本提示的遵循程度。
- **类别条件生成**：[[dit|DiT]] 在 ImageNet 类别条件生成中使用 CFG（最佳 scale=1.50），取得 FID 2.27 的 SOTA 结果[^src-dit]。
- **文本到视频/3D 生成**：扩散模型在视频和 3D 生成中同样采用 CFG 技术。
- **其他条件生成任务**：任何需要条件控制（如类别条件、布局条件）的扩散模型都可以使用 CFG。
- **外生条件时序概率预测**：[[kite|KITE]] 将 CFG 接到 Flow Matching 速度场上，条件为历史/未来外生变量；训练时以概率丢弃协变量集合，推理用 $\hat v=(1+\gamma)v(c)-\gamma v(\varnothing)$ 控制外生驱动强度（经验最优 $\gamma\approx 1.2$–$1.4$）。与 [[knowledge-guided-conditioning|KGC]]、[[history-conditional-manifold|HCM]] 的串联见 [[kite-manifold-guidance-chain]]。[^src-kite]
- **与 observation self-guidance 的区分**：[[tsdiff|TSDiff]] 的 [[observation-self-guidance|observation self-guidance]] **不**联合训练条件/无条件分支，也不做条件 dropout；它用无条件去噪网络的一步重构构造 $p(y_{\mathrm{obs}}\mid x_t)$ 引导项，属于“纯无条件训练 + 推理期自引导”，与 CFG 正交[^src-prs]。

## 动态 CFG（反馈引导）

标准 CFG 使用固定引导尺度 $w$（或 $\lambda$），但在高缺失率条件生成场景下（如时空插补），固定尺度会在条件信息不足时导致生成过程漂移到先验分布，远离条件观测[^src-fence]。[[feedback-diffusion-guidance|反馈扩散引导]] 和 [[fence|FENCE]] 将 $\lambda$ 变为去噪步 $k$ 和样本 $x_k$ 的函数[^src-fence]：

$$\lambda(x_k, k) \approx \frac{p_{\theta,k}(c|x_k)}{p_{\theta,k}(c|x_k) - (1-\pi)}$$

通过追踪扩散反向马尔可夫链估计后验似然 $p(c|x_k)$，实现引导尺度的动态调整——后验降低时增大引导，后验升高时减小引导[^src-fence]。

## 局限性

- **推理成本翻倍**：每步需两次模型前向传播（条件+无条件）。Ho & Salimans 建议后期注入条件可能缓解——但 $T=128$ 的 CFG（等价于 $T=256$ 的单次前传）FID 仍略逊于 ADM-G。[^src-classifier-free-diffusion-guidance]
- **高 $w$ 导致饱和伪影**：强引导下样本颜色过饱和、多样性急剧下降。在分布不平衡的应用中，低概率区域的样本可能被完全压制。[^src-classifier-free-diffusion-guidance]
- **固定尺度限制**：不同数据点对条件的满足程度不同，固定 $w$ 无法适配（[[fence|FENCE]] 通过动态反馈引导解决）。[^src-fence]

## LDM 中的应用

LDM 成功将无分类器引导应用于文本到图像生成[^src-rombach-ldm-2022]。在 MS-COCO 数据集上，CFG 将 FID 从 23.31 提升到 12.63（引导尺度 s=1.5）。典型引导尺度在 1.5 到 10.0 之间。[[instaflow|InstaFlow]] 为 Rectified Flow 设计了 CFG 等效机制 $v^\alpha = \alpha\cdot v(\cdot|T) + (1-\alpha)\cdot v(\cdot|\text{NULL})$，最佳 $\alpha\approx 1.5$ 远低于 SD 的 5-7.5[^src-instaflow]。

[^src-classifier-free-diffusion-guidance]: [[source-classifier-free-diffusion-guidance]]
[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
[^src-dit]: [[source-dit]]
[^src-instaflow]: [[source-instaflow]]
[^src-fence]: [[source-fence]]
[^src-kite]: [[source-kite]]
[^src-prs]: [[source-prs]]
