---
title: "TimeGrad"
type: entity
tags:
  - diffusion-models
  - time-series
  - probabilistic-forecasting
  - autoregressive
  - ddpm
  - icml-2021
created: 2026-05-31
last_updated: 2026-07-13
source_count: 9
confidence: medium
status: active
---

# TimeGrad

**TimeGrad** 是首个将[[ddpm|DDPM]]扩散模型应用于多变量时间序列概率预测的方法，由 Rasul, Seward, Schuster & Vollgraf（Zalando Research）发表于 ICML 2021[^src-timegrad]。其核心设计是"RNN 负责时间自回归编码 + 扩散模型负责每步多变量分布建模"的二段式架构，将扩散模型的生成能力从图像领域跨界引入时间序列预测。

## 核心创新

TimeGrad 的关键洞察在于：DDPM 的去噪网络 $\varepsilon_\theta$ 原本只以扩散步 $n$ 为条件——TimeGrad 将其条件扩展为 $(n, h_{t-1})$，其中 $h_{t-1}$ 是 RNN 从历史时序中提取的隐状态[^src-timegrad]：

$$\varepsilon_\theta(x_t^n, h_{t-1}, n) \quad\text{vs}\quad \varepsilon_\theta(x^n, n) \quad\text{(DDPM)}$$

这一改动意味着去噪网络在预测噪声的同时"感知"时序上下文——不是作为额外损失项，不是作为后处理，而是将条件自然地注入到去噪的每一步中[^src-timegrad]。这是能量基模型（EBM）的优雅体现：只需在回归目标里多加一个输入，联合分布就自动条件化。

## 架构

```
历史序列 x_{1:t-1}  + 协变量 c_{1:t}
        │
        ▼
   2层 LSTM (h=40)  ← 自回归时间编码器
        │
        ▼ h_{t-1}
   ┌────────────────────┐
   │  条件扩散模型        │
   │  ε_θ(x_t^n, h_{t-1}, n)
   │  8块膨胀卷积残差网络  │  ← WaveNet/DiffWave风格
   │  GAU: σ(·)⊙tanh(·)  │
   │  N=100步去噪采样      │
   └────────────────────┘
        │
        ▼
   x_t^0 (预测的多变量向量)
```

### RNN 编码器
2 层 LSTM（隐维度 40）将历史观测 $x_{t-1}^0$ 和协变量 $c_t$ 拼接后编码为 $h_{t-1}$[^src-timegrad]。协变量包含时间依赖嵌入（小时/天/星期）和时间无关嵌入（如序列 ID），类别特征使用嵌入层处理[^src-timegrad]。

### 条件扩散去噪网络 $\varepsilon_\theta$
基于 WaveNet/DiffWave 的膨胀卷积残差架构[^src-timegrad]：8 个残差块，每块由 1D 卷积（kernel=3）+ GAU（门控激活单元）$\sigma(W_f \cdot x) \odot \tanh(W_g \cdot x)$ 组成，残差通道=8，膨胀率交替 1 和 2。条件注入方式：$h_{t-1}$（40 维）和 $n$ 的 Transformer 正弦位置编码（32 维）通过全连接层变换后作为每层卷积的偏置项广播到 D 维[^src-timegrad]。总参数量极小（所有卷积通道仅 8）。

### 均值缩放归一化
每个维度 $i$ 的值除以其在上下文窗口内的均值（继承自 DeepAR），推理时乘以同一均值恢复原始尺度[^src-timegrad]。这使模型聚焦于"相对变化模式"而非"绝对数值范围"，在高维数据集上尤为关键。

## 训练

完全继承 DDPM 的 $L_{\text{simple}}$ 范式，是对每个时间步做噪声预测回归[^src-timegrad]：

对预测窗口的每个时间步 $t$：
1. 随机抽扩散步 $n \sim \text{Uniform}\{1,...,N=100\}$，随机抽噪声 $\varepsilon \sim \mathcal{N}(0, I)$
2. 构造加噪信号：$x_t^n = \sqrt{\bar\alpha_n}\, x_t^0 + \sqrt{1-\bar\alpha_n}\, \varepsilon$
3. 梯度下降：$\min_\theta \|\varepsilon - \varepsilon_\theta(x_t^n, h_{t-1}, n)\|^2$

噪声调度：$\beta_n$ 从 $10^{-4}$ 线性增加到 $0.1$（注意 $\beta_N=0.1$ 大于 DDPM 图像的 $0.02$，因时序残差分布范围更广）[^src-timegrad]。使用 Adam 优化器（lr=$10^{-3}$），batch size=64[^src-timegrad]。

## 推理

采用退火 Langevin 动力学采样，配合自回归外推[^src-timegrad]：

1. 训练集最后一段上下文窗口跑 RNN 得 $h_T$
2. 从白噪声 $x_{T+1}^N \sim \mathcal{N}(0,I)$ 出发，$N=100$ 步循环去噪：
   $$x_t^{n-1} = \frac{1}{\sqrt{\alpha_n}} \left[ x_t^n - \frac{\beta_n}{\sqrt{1-\bar\alpha_n}} \varepsilon_\theta(x_t^n, h_{t-1}, n) \right] + \sqrt{\tilde\beta_n}\, z$$
   其中 $z \sim \mathcal{N}(0,I)$ for $n>1$，$z=0$ for $n=1$（最后一步确定性输出）
3. 将采样得到的 $x_{T+1}^0$ 送回 RNN 得 $h_{T+1}$，重复至预测窗口结束
4. 整个过程重复 $S=100$ 次得到 100 条轨迹，计算经验 CDF 后用 CRPS 评估

## 性能

在 6 个真实数据集（Exchange D=8, Solar D=137, Electricity D=370, Traffic D=963, Taxi D=1214, Wikipedia D=2000）上以 CRPS_sum 评估，TimeGrad 在 5 个数据集上排名第一，14 种基线全面领先[^src-timegrad]。

消融实验揭示了一个与图像扩散截然不同的规律：$N \approx 10$ 时 CRPS_sum 已接近最优，$N=100$ 达最优，$N>100$ 后性能持平甚至轻微下降[^src-timegrad]。原因在于 RNN 隐状态 $h_{t-1}$ 已提供了强引导信号——扩散模型只需补充建模残差中的不确定性，无需从零重建完整数据分布[^src-timegrad]。这打破了"扩散一定需要上千步"的刻板印象。

## 与相关方法的关系

TimeGrad 处于多条研究线的交汇点：

- **继承自 [[ddpm|DDPM]]**：$\varepsilon$ 预测参数化、$L_{\text{simple}}$ 简化目标、$\beta$ 调度、$\bar\alpha_t$ 累积积、N 步 Markov 链——全部直接搬用；区别仅在于条件扩展为 $(n, h_{t-1})$[^src-timegrad]
- **继承自 DeepAR**：自回归 RNN 架构和均值缩放归一化，但将预设参数化输出分布替换为更灵活的扩散模型[^src-timegrad]
- **对标 Vec-LSTM**：Vec-LSTM 用低秩高斯 Copula 建模跨维度依赖，TimeGrad 用扩散模型隐式学习任意复杂度的联合分布[^src-timegrad]
- **后续推动**：[[generative-time-series-forecasting|生成式时间序列预测]]整个方向的奠基之作——[[csdi|CSDI]]（插补）、[[tsdiff|TSDiff]]（无条件 + observation self-guidance，NeurIPS 2023）[^src-prs]、DiffSTG（时空图）、[[specstg|SpecSTG]]（谱域扩散）、[[urbandit|UrbanDiT]]（扩散 Transformer）均以 TimeGrad 的条件扩散范式为理论起点或对照基线

## 局限性

1. **推理速度是根本瓶颈**：预测 24 步需 $100 \times 24 = 2400$ 次 $\varepsilon_\theta$ 前向传播，实时场景不可行[^src-timegrad]
2. **自回归串行依赖**：不同时间步必须串行推理，无法并行预测[^src-timegrad]
3. **固定 LSTM 瓶颈**：D=8 到 D=2000 共用 h=40，高维信息压缩比不足[^src-timegrad]
4. **统一噪声调度**：所有数据集使用相同 $\beta$ 调度，CSDI 后来将其改进为可学习调度[^src-timegrad]
5. **缺少显式空间/拓扑归纳偏置**：跨维度依赖依赖扩散模型容量黑箱学习，当存在明确图拓扑时不如 GNN 方法高效[^src-timegrad]

## 关联页面

- [[ddpm]] — DDPM，TimeGrad 的扩散模型基础
- [[diffusion-model]] — 扩散模型概念总览
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[simdiff]] — SimDiff，端到端扩散时间序列点预测（AAAI 2026）
- [[specstg]] — SpecSTG，谱域扩散时空图预测（arXiv 2024）
- [[tedm]] — TEDM，EDM 框架的 O(H) 采样时序扩散模型（ICLR 2026）
- [[urbandit]] — UrbanDiT，扩散 Transformer 时空基础模型（NeurIPS 2025）
- [[nsdiff]] — NsDiff (ICML 2025 Spotlight)，以 LSNM+UANS 超越 TimeGrad 的时序扩散 SOTA[^src-nsdiff]
- [[stats]] — StaTS (arXiv 2026)，联合频谱轨迹调度学习+频率引导去噪，将固定调度推进到自适应调度[^src-stats]
- [[s2dbm]] — S²DBM (arXiv 2024)，用[[brownian-bridge-diffusion|布朗桥]]把扩散过程两端钉住、以 s=0 实现无噪声确定性采样，在点对点预测上系统性超越 TimeGrad 范式的含噪条件扩散[^src-s2dbm]
- [[ratd|RATD]] (NeurIPS 2024)，检索增强的时序扩散模型——从数据库检索最近邻参照引导去噪，非自回归采样使其速度甚至略优于 TimeGrad 的自回归解码[^src-ratd]
- [[probts|ProbTS]]（NeurIPS 2024）将 TimeGrad 作为 **AR 概率预测** 代表：短程分布估计强，但长程 CRPS 随 horizon/趋势恶化（误差累积）；在强季节（如 Traffic）上 AR 可反超 PatchTST；均值缩放是短程默认可靠归一化，RevIN 对长程 AR 更有帮助[^src-probts]
- [[armd|ARMD]]（AAAI 2025）将 TimeGrad 这类"把真实序列扩散成白高斯噪声、再以历史为条件去噪"的范式视为扩散机制与 TSF 目标的失配，转而用滑动窗口产生确定性中间态、把历史→未来直接构造成扩散链，并在 7 个数据集上显著超越 TimeGrad[^src-armd]
- [[tsdiff|TSDiff]]（NeurIPS 2023）改走**无条件**训练 + 推理期 [[observation-self-guidance|observation self-guidance]]，在相关工作中将 TimeGrad 定位为条件扩散预测代表，并与 CSDI/SSSD 对照任务专用性[^src-prs]
- [[manf|MANF]]（arXiv 2022）走 **NAR + 条件 RealNVP** 路线，与 TimeGrad 的 **AR + 条件扩散** 形成对照：同属高维联合分布灵活建模，但解码与生成机制不同；MANF 基线中含 Transformer-MAF 等 AR 流，而 TimeGrad 报告中 Transformer-MAF 为强竞争方法[^src-maf]

[^src-timegrad]: [[source-timegrad]]
[^src-nsdiff]: [[source-nsdiff]]
[^src-stats]: [[source-stats]]
[^src-s2dbm]: [[source-s2dbm]]
[^src-ratd]: [[source-ratd]]
[^src-armd]: [[source-armd]]
[^src-probts]: [[source-probts]]
[^src-prs]: [[source-prs]]
[^src-maf]: [[source-maf]]
