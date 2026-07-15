---
title: "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - probabilistic-imputation
  - self-supervised-learning
  - neurips-2021
created: 2026-05-31
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Source: CSDI

**作者**: Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon (Stanford University; Mitsubishi UFJ Trust Investment Technology Institute; Japan Digital Design)
**发表**: NeurIPS 2021
**领域**: 多元时间序列概率插补

## 核心论点

CSDI 是首个将条件扩散模型显式用于时间序列缺失值插补的工作[^src-csdi]。此前利用扩散模型做插补的方法（Song et al., 2021; Kadkhodaie & Simoncelli, 2021）采用"事后修补"策略——用预训练的无条件扩散模型，在采样时将已知观测值硬注入生成结果。CSDI 的洞察是：既然目标是对条件分布 $p(x^{\text{ta}} \mid x^{\text{co}})$ 采样，就应该在训练阶段直接让去噪网络学习这个条件分布，而非事后近似。为此 CSDI 做了三件事：(1) 将 [[ddpm|DDPM]] 的去噪函数 $\epsilon_\theta(x_t, t)$ 扩展为条件形式 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$；(2) 设计自监督训练策略，从观测值中人工构造"伪缺失目标"和"伪条件观测"；(3) 用时间 Transformer + 特征 Transformer 的双轴注意力替代 DiffWave 的膨胀卷积，分别捕获时间依赖和跨特征依赖[^src-csdi]。

## 方法

- **条件扩散**：将 DDPM 的参数化 $\mu_\theta(x_t, t)$ 直接扩展到条件场景 $\mu_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$，训练目标保持 MSE 噪声预测，唯一区别是 $\epsilon_\theta$ 多了条件输入 $x_0^{\text{co}}$。通过零填充将变化的输入形状固定为 $K \times L$，配合条件掩码 $m^{\text{co}}$ 指示条件观测位置[^src-csdi]
- **自监督训练**：受 BERT 掩码语言建模启发，从训练样本的观测值中随机选取一部分作为"伪插补目标" $x_0^{\text{ta}}$，其余作为"伪条件观测" $x_0^{\text{co}}$，在伪目标上按标准扩散流程加噪后训练去噪网络。四种目标选择策略：Random（采样 0-100% 观测值为目标，应对未知缺失）、Historical（借用训练集另一样本的缺失模式，应对结构化缺失如传感器连续故障块）、Mix（Random + Historical 各 50%）、Test pattern（已知测试缺失模式时直接使用，如预测任务）。详见 [[self-supervised-imputation-training|自监督插补训练]][^src-csdi]
- **双轴注意力架构**：每层残差块包含两个 1 层 Transformer 编码器——时间 Transformer 沿特征轴学习时间依赖，特征 Transformer 沿时间轴学习跨特征依赖。骨架为 DiffWave 风格 4 层残差层，残差通道 $C=64$，8 注意力头。输入侧信息：128 维扩散步嵌入（正弦）、128 维时间戳嵌入、16 维特征类别嵌入。输出经 $(1-m^{\text{co}})$ 掩码屏蔽条件位置。$T=50$，二次方噪声调度 $\beta_1=0.0001, \beta_T=0.5$，约 415K 参数[^src-csdi]

## 关键结果

- **概率插补**：在 PhysioNet 医疗数据（35 特征，48 时间步，~80% 缺失率）和北京空气质量数据（36 特征，36 时间步，~13% 结构性缺失）上对比 Multitask GP、GP-VAE、V-RIN 和无条件扩散基线。CSDI 的 CRPS 比 GP-VAE 降低 40-65%；无条件扩散本身已优于 GP-VAE（PhysioNet 50%：0.458 vs 0.774），CSDI 条件建模进一步带来约 28% 额外改善[^src-csdi]
- **确定性插补**：中位数聚合 100 个生成样本，MAE 比 BRITS/GLIMA/RDIS 降低 5-20%。缺失率越低 CSDI 相对优势越大（更多观测值提供更丰富条件信息）[^src-csdi]
- **不规则采样插值**：CRPS 大幅领先 Latent ODE 和 mTANs（PhysioNet 50% 缺失：CSDI 0.418 vs mTANs 0.567 vs Latent ODE 0.676），验证了注意力机制对不规则采样的天然适配[^src-csdi]
- **预测**：在 5 个 GluonTS 基准（solar/electricity/traffic/taxi/wiki）上 CRPS-sum 在 electricity 和 traffic 上超越 [[timegrad|TimeGrad]]（0.017 vs 0.021；0.020 vs 0.044），整体竞争力相当；预测优势不如插补显著——预测集几乎没有缺失值，RNN 不受限[^src-csdi]
- **消融**：移除时间或特征注意力均显著降性能；Bi-RNN/膨胀卷积替代劣于双轴注意力；噪声调度对比表明 CRPS 对不同调度稳健，但 ELBO/NLL 高度依赖调度——不可靠用于评估生成质量；5-10 个样本即接近最优，超过 50 个改善趋于饱和[^src-csdi]

## 贡献

1. 开创了"条件扩散模型 + 自监督训练 + 时间序列插补"的方向，证明条件建模相比无条件扩散+事后约束有实质性收益（实验差值约 28% CRPS 改善）
2. 证明 T=50 步扩散对时间序列插补足够，远低于图像的 T=1000
3. 为后续工作（SSSD、[[tsdiff|TSDiff]] 中的条件对照、CSDI 变体等）奠定了标准范式；TSDiff 在预测基准上直接与 CSDI 比较，并强调无条件 + 推理引导的任务无关替代路线

## 局限性

- **计算效率**：T=50 步仍需串行推理，实时场景受限
- **注意力 $O(L^2)$ 复杂度**：长序列时计算成本膨胀（后续 SSSD 用 S4 状态空间模型替代 Transformer 作为回应）
- **假设观测值可靠**：未处理传感器故障导致的错误观测值被当作条件注入的风险
- **仅验证时间序列**：声称框架不限于时序，但未在表格数据或图像修复上验证

[^src-csdi]: [[source-csdi]]
